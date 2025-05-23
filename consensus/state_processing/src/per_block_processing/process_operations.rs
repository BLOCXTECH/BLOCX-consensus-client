use super::*;
use crate::common::{
    get_attestation_participation_flag_indices, increase_balance, initiate_validator_exit,
    slash_validator,
};
use crate::per_block_processing::errors::{BlockProcessingError, IntoWithIndex};
use crate::VerifySignatures;
use types::consts::altair::{PARTICIPATION_FLAG_WEIGHTS, PROPOSER_WEIGHT, WEIGHT_DENOMINATOR};
use types::typenum::U33;
use types::validator::is_compounding_withdrawal_credential;

pub fn process_operations<E: EthSpec, Payload: AbstractExecPayload<E>>(
    state: &mut BeaconState<E>,
    block_body: BeaconBlockBodyRef<E, Payload>,
    verify_signatures: VerifySignatures,
    ctxt: &mut ConsensusContext<E>,
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    process_proposer_slashings(
        state,
        block_body.proposer_slashings(),
        verify_signatures,
        ctxt,
        spec,
    )?;
    process_attester_slashings(
        state,
        block_body.attester_slashings(),
        verify_signatures,
        ctxt,
        spec,
    )?;
    process_attestations(state, block_body, verify_signatures, ctxt, spec)?;
    process_deposits(state, block_body.deposits(), spec)?;
    process_exits(state, block_body.voluntary_exits(), verify_signatures, spec)?;

    if let Ok(bls_to_execution_changes) = block_body.bls_to_execution_changes() {
        process_bls_to_execution_changes(state, bls_to_execution_changes, verify_signatures, spec)?;
    }

    if state.fork_name_unchecked().electra_enabled() {
        state.update_pubkey_cache()?;
        process_deposit_requests(state, &block_body.execution_requests()?.deposits, spec)?;
        process_withdrawal_requests(state, &block_body.execution_requests()?.withdrawals, spec)?;
        process_consolidation_requests(
            state,
            &block_body.execution_requests()?.consolidations,
            spec,
        )?;
    }

    Ok(())
}

pub mod base {
    use super::*;

    /// Validates each `Attestation` and updates the state, short-circuiting on an invalid object.
    ///
    /// Returns `Ok(())` if the validation and state updates completed successfully, otherwise returns
    /// an `Err` describing the invalid object or cause of failure.
    pub fn process_attestations<'a, E: EthSpec, I>(
        state: &mut BeaconState<E>,
        attestations: I,
        verify_signatures: VerifySignatures,
        ctxt: &mut ConsensusContext<E>,
        spec: &ChainSpec,
    ) -> Result<(), BlockProcessingError>
    where
        I: Iterator<Item = AttestationRef<'a, E>>,
    {
        // Ensure required caches are all built. These should be no-ops during regular operation.
        state.build_committee_cache(RelativeEpoch::Current, spec)?;
        state.build_committee_cache(RelativeEpoch::Previous, spec)?;
        initialize_epoch_cache(state, spec)?;
        initialize_progressive_balances_cache(state, spec)?;
        state.build_slashings_cache()?;

        let proposer_index = ctxt.get_proposer_index(state, spec)?;

        // Verify and apply each attestation.
        for (i, attestation) in attestations.enumerate() {
            verify_attestation_for_block_inclusion(
                state,
                attestation,
                ctxt,
                verify_signatures,
                spec,
            )
            .map_err(|e| e.into_with_index(i))?;

            let AttestationRef::Base(attestation) = attestation else {
                // Pending attestations have been deprecated in a altair, this branch should
                // never happen
                return Err(BlockProcessingError::PendingAttestationInElectra);
            };

            let pending_attestation = PendingAttestation {
                aggregation_bits: attestation.aggregation_bits.clone(),
                data: attestation.data.clone(),
                inclusion_delay: state.slot().safe_sub(attestation.data.slot)?.as_u64(),
                proposer_index,
            };

            if attestation.data.target.epoch == state.current_epoch() {
                state
                    .as_base_mut()?
                    .current_epoch_attestations
                    .push(pending_attestation)?;
            } else {
                state
                    .as_base_mut()?
                    .previous_epoch_attestations
                    .push(pending_attestation)?;
            }
        }

        Ok(())
    }
}

pub mod altair_deneb {
    use super::*;
    use crate::common::update_progressive_balances_cache::update_progressive_balances_on_attestation;

    pub fn process_attestations<'a, E: EthSpec, I>(
        state: &mut BeaconState<E>,
        attestations: I,
        verify_signatures: VerifySignatures,
        ctxt: &mut ConsensusContext<E>,
        spec: &ChainSpec,
    ) -> Result<(), BlockProcessingError>
    where
        I: Iterator<Item = AttestationRef<'a, E>>,
    {
        attestations.enumerate().try_for_each(|(i, attestation)| {
            process_attestation(state, attestation, i, ctxt, verify_signatures, spec)
        })
    }

    pub fn process_attestation<E: EthSpec>(
        state: &mut BeaconState<E>,
        attestation: AttestationRef<E>,
        att_index: usize,
        ctxt: &mut ConsensusContext<E>,
        verify_signatures: VerifySignatures,
        spec: &ChainSpec,
    ) -> Result<(), BlockProcessingError> {
        let _proposer_index = ctxt.get_proposer_index(state, spec)?;
        let previous_epoch = ctxt.previous_epoch;
        let current_epoch = ctxt.current_epoch;

        let indexed_att = verify_attestation_for_block_inclusion(
            state,
            attestation,
            ctxt,
            verify_signatures,
            spec,
        )
        .map_err(|e| e.into_with_index(att_index))?;

        // Only update participation flags without any reward processing
        // All rewards will be handled centrally in per_block_processing.rs
        let data = attestation.data();
        let inclusion_delay = state.slot().safe_sub(data.slot)?.as_u64();
        let participation_flag_indices =
            get_attestation_participation_flag_indices(state, data, inclusion_delay, spec)?;

        // Update epoch participation flags without applying any rewards
        for index in indexed_att.attesting_indices_iter() {
            let index = *index as usize;

            for (flag_index, _) in PARTICIPATION_FLAG_WEIGHTS.iter().enumerate() {
                let epoch_participation = state.get_epoch_participation_mut(
                    data.target.epoch,
                    previous_epoch,
                    current_epoch,
                )?;

                if participation_flag_indices.contains(&flag_index) {
                    let validator_participation = epoch_participation
                        .get_mut(index)
                        .ok_or(BeaconStateError::ParticipationOutOfBounds(index))?;

                    if !validator_participation.has_flag(flag_index)? {
                        validator_participation.add_flag(flag_index)?;
                        
                        // Only update the cache, but DO NOT apply any rewards
                        let validator_effective_balance = state.epoch_cache().get_effective_balance(index)?;
                        let validator_slashed = state.slashings_cache().is_slashed(index);
                        
                        update_progressive_balances_on_attestation(
                            state,
                            data.target.epoch,
                            flag_index,
                            validator_effective_balance,
                            validator_slashed,
                        )?;
                    }
                }
            }
        }
        
        // NO REWARD IS APPLIED HERE - all rewards are managed centrally in per_block_processing.rs
        
        Ok(())
    }
}

/// Validates each `ProposerSlashing` and updates the state, short-circuiting on an invalid object.
///
/// Returns `Ok(())` if the validation and state updates completed successfully, otherwise returns
/// an `Err` describing the invalid object or cause of failure.
pub fn process_proposer_slashings<E: EthSpec>(
    state: &mut BeaconState<E>,
    proposer_slashings: &[ProposerSlashing],
    verify_signatures: VerifySignatures,
    ctxt: &mut ConsensusContext<E>,
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    state.build_slashings_cache()?;

    // Verify and apply proposer slashings in series.
    // We have to verify in series because an invalid block may contain multiple slashings
    // for the same validator, and we need to correctly detect and reject that.
    proposer_slashings
        .iter()
        .enumerate()
        .try_for_each(|(i, proposer_slashing)| {
            verify_proposer_slashing(proposer_slashing, state, verify_signatures, spec)
                .map_err(|e| e.into_with_index(i))?;

            slash_validator(
                state,
                proposer_slashing.signed_header_1.message.proposer_index as usize,
                None,
                ctxt,
                spec,
            )?;

            Ok(())
        })
}

/// Validates each `AttesterSlashing` and updates the state, short-circuiting on an invalid object.
///
/// Returns `Ok(())` if the validation and state updates completed successfully, otherwise returns
/// an `Err` describing the invalid object or cause of failure.
pub fn process_attester_slashings<'a, E: EthSpec, I>(
    state: &mut BeaconState<E>,
    attester_slashings: I,
    verify_signatures: VerifySignatures,
    ctxt: &mut ConsensusContext<E>,
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError>
where
    I: Iterator<Item = AttesterSlashingRef<'a, E>>,
{
    state.build_slashings_cache()?;

    for (i, attester_slashing) in attester_slashings.enumerate() {
        let slashable_indices =
            verify_attester_slashing(state, attester_slashing, verify_signatures, spec)
                .map_err(|e| e.into_with_index(i))?;

        for i in slashable_indices {
            slash_validator(state, i as usize, None, ctxt, spec)?;
        }
    }

    Ok(())
}

/// Wrapper function to handle calling the correct version of `process_attestations` based on
/// the fork.
pub fn process_attestations<E: EthSpec, Payload: AbstractExecPayload<E>>(
    state: &mut BeaconState<E>,
    block_body: BeaconBlockBodyRef<E, Payload>,
    verify_signatures: VerifySignatures,
    ctxt: &mut ConsensusContext<E>,
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    match block_body {
        BeaconBlockBodyRef::Base(_) => {
            base::process_attestations(
                state,
                block_body.attestations(),
                verify_signatures,
                ctxt,
                spec,
            )?;
        }
        BeaconBlockBodyRef::Altair(_)
        | BeaconBlockBodyRef::Bellatrix(_)
        | BeaconBlockBodyRef::Capella(_)
        | BeaconBlockBodyRef::Deneb(_)
        | BeaconBlockBodyRef::Electra(_) => {
            altair_deneb::process_attestations(
                state,
                block_body.attestations(),
                verify_signatures,
                ctxt,
                spec,
            )?;
        }
    }
    Ok(())
}

/// Validates each `Exit` and updates the state, short-circuiting on an invalid object.
///
/// Returns `Ok(())` if the validation and state updates completed successfully, otherwise returns
/// an `Err` describing the invalid object or cause of failure.
pub fn process_exits<E: EthSpec>(
    state: &mut BeaconState<E>,
    voluntary_exits: &[SignedVoluntaryExit],
    verify_signatures: VerifySignatures,
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    // Verify and apply each exit in series. We iterate in series because higher-index exits may
    // become invalid due to the application of lower-index ones.
    for (i, exit) in voluntary_exits.iter().enumerate() {
        verify_exit(state, None, exit, verify_signatures, spec)
            .map_err(|e| e.into_with_index(i))?;

        initiate_validator_exit(state, exit.message.validator_index as usize, spec)?;
    }
    Ok(())
}

/// Validates each `bls_to_execution_change` and updates the state
///
/// Returns `Ok(())` if the validation and state updates completed successfully. Otherwise returns
/// an `Err` describing the invalid object or cause of failure.
pub fn process_bls_to_execution_changes<E: EthSpec>(
    state: &mut BeaconState<E>,
    bls_to_execution_changes: &[SignedBlsToExecutionChange],
    verify_signatures: VerifySignatures,
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    for (i, signed_address_change) in bls_to_execution_changes.iter().enumerate() {
        verify_bls_to_execution_change(state, signed_address_change, verify_signatures, spec)
            .map_err(|e| e.into_with_index(i))?;

        state
            .get_validator_mut(signed_address_change.message.validator_index as usize)?
            .change_withdrawal_credentials(
                &signed_address_change.message.to_execution_address,
                spec,
            );
    }

    Ok(())
}

/// Validates each `Deposit` and updates the state, short-circuiting on an invalid object.
///
/// Returns `Ok(())` if the validation and state updates completed successfully, otherwise returns
/// an `Err` describing the invalid object or cause of failure.
pub fn process_deposits<E: EthSpec>(
    state: &mut BeaconState<E>,
    deposits: &[Deposit],
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    // [Modified in Electra:EIP6110]
    // Disable former deposit mechanism once all prior deposits are processed
    let deposit_requests_start_index = state.deposit_requests_start_index().unwrap_or(u64::MAX);
    let eth1_deposit_index_limit = std::cmp::min(
        deposit_requests_start_index,
        state.eth1_data().deposit_count,
    );

    if state.eth1_deposit_index() < eth1_deposit_index_limit {
        let expected_deposit_len = std::cmp::min(
            E::MaxDeposits::to_u64(),
            state.get_outstanding_deposit_len()?,
        );
        block_verify!(
            deposits.len() as u64 == expected_deposit_len,
            BlockProcessingError::DepositCountInvalid {
                expected: expected_deposit_len as usize,
                found: deposits.len(),
            }
        );
    } else {
        block_verify!(
            deposits.len() as u64 == 0,
            BlockProcessingError::DepositCountInvalid {
                expected: 0,
                found: deposits.len(),
            }
        );
    }

    // Verify merkle proofs in parallel.
    deposits
        .par_iter()
        .enumerate()
        .try_for_each(|(i, deposit)| {
            verify_deposit_merkle_proof(
                state,
                deposit,
                state.eth1_deposit_index().safe_add(i as u64)?,
                spec,
            )
            .map_err(|e| e.into_with_index(i))
        })?;

    // Update the state in series.
    for deposit in deposits {
        apply_deposit(state, deposit.data.clone(), None, true, spec)?;
    }

    Ok(())
}

/// Process a single deposit, verifying its merkle proof if provided.
pub fn apply_deposit<E: EthSpec>(
    state: &mut BeaconState<E>,
    deposit_data: DepositData,
    proof: Option<FixedVector<Hash256, U33>>,
    increment_eth1_deposit_index: bool,
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    let deposit_index = state.eth1_deposit_index() as usize;
    if let Some(proof) = proof {
        let deposit = Deposit {
            proof,
            data: deposit_data.clone(),
        };
        verify_deposit_merkle_proof(state, &deposit, state.eth1_deposit_index(), spec)
            .map_err(|e| e.into_with_index(deposit_index))?;
    }

    if increment_eth1_deposit_index {
        state.eth1_deposit_index_mut().safe_add_assign(1)?;
    }

    // Get an `Option<u64>` where `u64` is the validator index if this deposit public key
    // already exists in the beacon_state.
    let validator_index = get_existing_validator_index(state, &deposit_data.pubkey)
        .map_err(|e| e.into_with_index(deposit_index))?;

    let amount = deposit_data.amount;

    if let Some(index) = validator_index {
        // [Modified in Electra:EIP7251]
        if let Ok(pending_balance_deposits) = state.pending_balance_deposits_mut() {
            pending_balance_deposits.push(PendingBalanceDeposit { index, amount })?;

            let validator = state
                .validators()
                .get(index as usize)
                .ok_or(BeaconStateError::UnknownValidator(index as usize))?;

            if is_compounding_withdrawal_credential(deposit_data.withdrawal_credentials, spec)
                && validator.has_eth1_withdrawal_credential(spec)
                && is_valid_deposit_signature(&deposit_data, spec).is_ok()
            {
                state.switch_to_compounding_validator(index as usize, spec)?;
            }
        } else {
            // Update the existing validator balance.
            increase_balance(state, index as usize, amount)?;
        }
    } else {
        // The signature should be checked for new validators. Return early for a bad
        // signature.
        if is_valid_deposit_signature(&deposit_data, spec).is_err() {
            return Ok(());
        }

        state.add_validator_to_registry(&deposit_data, spec)?;
        let new_validator_index = state.validators().len().safe_sub(1)? as u64;

        // [New in Electra:EIP7251]
        if let Ok(pending_balance_deposits) = state.pending_balance_deposits_mut() {
            pending_balance_deposits.push(PendingBalanceDeposit {
                index: new_validator_index,
                amount,
            })?;
        }
    }

    Ok(())
}

// Make sure to build the pubkey cache before calling this function
pub fn process_withdrawal_requests<E: EthSpec>(
    state: &mut BeaconState<E>,
    requests: &[WithdrawalRequest],
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    for request in requests {
        let amount = request.amount;
        let is_full_exit_request = amount == spec.full_exit_request_amount;

        // If partial withdrawal queue is full, only full exits are processed
        if state.pending_partial_withdrawals()?.len() == E::pending_partial_withdrawals_limit()
            && !is_full_exit_request
        {
            continue;
        }

        // Verify pubkey exists
        let Some(index) = state.pubkey_cache().get(&request.validator_pubkey) else {
            continue;
        };

        let validator = state.get_validator(index)?;
        // Verify withdrawal credentials
        let has_correct_credential = validator.has_execution_withdrawal_credential(spec);
        let is_correct_source_address = validator
            .get_execution_withdrawal_address(spec)
            .map(|addr| addr == request.source_address)
            .unwrap_or(false);

        if !(has_correct_credential && is_correct_source_address) {
            continue;
        }

        // Verify the validator is active
        if !validator.is_active_at(state.current_epoch()) {
            continue;
        }

        // Verify exit has not been initiated
        if validator.exit_epoch != spec.far_future_epoch {
            continue;
        }

        // Verify the validator has been active long enough
        if state.current_epoch()
            < validator
                .activation_epoch
                .safe_add(spec.shard_committee_period)?
        {
            continue;
        }

        let pending_balance_to_withdraw = state.get_pending_balance_to_withdraw(index)?;
        if is_full_exit_request {
            // Only exit validator if it has no pending withdrawals in the queue
            if pending_balance_to_withdraw == 0 {
                initiate_validator_exit(state, index, spec)?
            }
            continue;
        }

        let balance = state.get_balance(index)?;
        let has_sufficient_effective_balance =
            validator.effective_balance >= spec.min_activation_balance;
        let has_excess_balance = balance
            > spec
                .min_activation_balance
                .safe_add(pending_balance_to_withdraw)?;

        // Only allow partial withdrawals with compounding withdrawal credentials
        if validator.has_compounding_withdrawal_credential(spec)
            && has_sufficient_effective_balance
            && has_excess_balance
        {
            let to_withdraw = std::cmp::min(
                balance
                    .safe_sub(spec.min_activation_balance)?
                    .safe_sub(pending_balance_to_withdraw)?,
                amount,
            );
            let exit_queue_epoch = state.compute_exit_epoch_and_update_churn(to_withdraw, spec)?;
            let withdrawable_epoch =
                exit_queue_epoch.safe_add(spec.min_validator_withdrawability_delay)?;
            state
                .pending_partial_withdrawals_mut()?
                .push(PendingPartialWithdrawal {
                    index: index as u64,
                    amount: to_withdraw,
                    withdrawable_epoch,
                })?;
        }
    }
    Ok(())
}

pub fn process_deposit_requests<E: EthSpec>(
    state: &mut BeaconState<E>,
    deposit_requests: &[DepositRequest],
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    for request in deposit_requests {
        // Set deposit receipt start index
        if state.deposit_requests_start_index()? == spec.unset_deposit_requests_start_index {
            *state.deposit_requests_start_index_mut()? = request.index
        }
        let deposit_data = DepositData {
            pubkey: request.pubkey,
            withdrawal_credentials: request.withdrawal_credentials,
            amount: request.amount,
            signature: request.signature.clone().into(),
        };
        apply_deposit(state, deposit_data, None, false, spec)?
    }

    Ok(())
}

// Make sure to build the pubkey cache before calling this function
pub fn process_consolidation_requests<E: EthSpec>(
    state: &mut BeaconState<E>,
    consolidation_requests: &[ConsolidationRequest],
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    for request in consolidation_requests {
        process_consolidation_request(state, request, spec)?;
    }

    Ok(())
}

pub fn process_consolidation_request<E: EthSpec>(
    state: &mut BeaconState<E>,
    consolidation_request: &ConsolidationRequest,
    spec: &ChainSpec,
) -> Result<(), BlockProcessingError> {
    // If the pending consolidations queue is full, consolidation requests are ignored
    if state.pending_consolidations()?.len() == E::PendingConsolidationsLimit::to_usize() {
        return Ok(());
    }
    // If there is too little available consolidation churn limit, consolidation requests are ignored
    if state.get_consolidation_churn_limit(spec)? <= spec.min_activation_balance {
        return Ok(());
    }

    let Some(source_index) = state
        .pubkey_cache()
        .get(&consolidation_request.source_pubkey)
    else {
        // source validator doesn't exist
        return Ok(());
    };
    let Some(target_index) = state
        .pubkey_cache()
        .get(&consolidation_request.target_pubkey)
    else {
        // target validator doesn't exist
        return Ok(());
    };
    // Verify that source != target, so a consolidation cannot be used as an exit.
    if source_index == target_index {
        return Ok(());
    }

    let source_validator = state.get_validator(source_index)?;
    // Verify the source withdrawal credentials
    if let Some(withdrawal_address) = source_validator.get_execution_withdrawal_address(spec) {
        if withdrawal_address != consolidation_request.source_address {
            return Ok(());
        }
    } else {
        // Source doen't have execution withdrawal credentials
        return Ok(());
    }

    let target_validator = state.get_validator(target_index)?;
    // Verify the target has execution withdrawal credentials
    if !target_validator.has_execution_withdrawal_credential(spec) {
        return Ok(());
    }

    // Verify the source and target are active
    let current_epoch = state.current_epoch();
    if !source_validator.is_active_at(current_epoch)
        || !target_validator.is_active_at(current_epoch)
    {
        return Ok(());
    }
    // Verify exits for source and target have not been initiated
    if source_validator.exit_epoch != spec.far_future_epoch
        || target_validator.exit_epoch != spec.far_future_epoch
    {
        return Ok(());
    }

    // Initiate source validator exit and append pending consolidation
    let source_exit_epoch = state
        .compute_consolidation_epoch_and_update_churn(source_validator.effective_balance, spec)?;
    let source_validator = state.get_validator_mut(source_index)?;
    source_validator.exit_epoch = source_exit_epoch;
    source_validator.withdrawable_epoch =
        source_exit_epoch.safe_add(spec.min_validator_withdrawability_delay)?;
    state
        .pending_consolidations_mut()?
        .push(PendingConsolidation {
            source_index: source_index as u64,
            target_index: target_index as u64,
        })?;

    Ok(())
}