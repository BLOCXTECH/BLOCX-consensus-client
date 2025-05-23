use crate::consts::altair::{
    SYNC_COMMITTEE_SUBNET_COUNT, TARGET_AGGREGATORS_PER_SYNC_SUBCOMMITTEE,
};
use crate::{
    ChainSpec, Domain, EthSpec, Fork, Hash256, PublicKey, SecretKey, Signature, SignedRoot, Slot,
    SyncAggregatorSelectionData,
};
use ethereum_hashing::hash;
use safe_arith::{ArithError, SafeArith};
use ssz::Encode;
use ssz_types::typenum::Unsigned;
use std::cmp;

#[derive(arbitrary::Arbitrary, PartialEq, Debug, Clone)]
pub struct SyncSelectionProof(Signature);

impl SyncSelectionProof {
    pub fn new<E: EthSpec>(
        slot: Slot,
        subcommittee_index: u64,
        secret_key: &SecretKey,
        fork: &Fork,
        genesis_validators_root: Hash256,
        spec: &ChainSpec,
    ) -> Self {
        let domain = spec.get_domain(
            slot.epoch(E::slots_per_epoch()),
            Domain::SyncCommitteeSelectionProof,
            fork,
            genesis_validators_root,
        );
        let message = SyncAggregatorSelectionData {
            slot,
            subcommittee_index,
        }
        .signing_root(domain);

        Self(secret_key.sign(message))
    }

    /// Returns the "modulo" used for determining if a `SyncSelectionProof` elects an aggregator.
    pub fn modulo<E: EthSpec>() -> Result<u64, ArithError> {
        Ok(cmp::max(
            1,
            (E::SyncCommitteeSize::to_u64())
                .safe_div(SYNC_COMMITTEE_SUBNET_COUNT)?
                .safe_div(TARGET_AGGREGATORS_PER_SYNC_SUBCOMMITTEE)?,
        ))
    }

    pub fn is_aggregator<E: EthSpec>(&self) -> Result<bool, ArithError> {
        self.is_aggregator_from_modulo(Self::modulo::<E>()?)
    }

    pub fn is_aggregator_from_modulo(&self, modulo: u64) -> Result<bool, ArithError> {
        let signature_hash = hash(&self.0.as_ssz_bytes());
        let signature_hash_int = u64::from_le_bytes(
            signature_hash
                .get(0..8)
                .expect("hash is 32 bytes")
                .try_into()
                .expect("first 8 bytes of signature should always convert to fixed array"),
        );

        signature_hash_int.safe_rem(modulo).map(|rem| rem == 0)
    }

    pub fn verify<E: EthSpec>(
        &self,
        slot: Slot,
        subcommittee_index: u64,
        pubkey: &PublicKey,
        fork: &Fork,
        genesis_validators_root: Hash256,
        spec: &ChainSpec,
    ) -> bool {
        let domain = spec.get_domain(
            slot.epoch(E::slots_per_epoch()),
            Domain::SyncCommitteeSelectionProof,
            fork,
            genesis_validators_root,
        );
        let message = SyncAggregatorSelectionData {
            slot,
            subcommittee_index,
        }
        .signing_root(domain);

        self.0.verify(pubkey, message)
    }
}

impl From<SyncSelectionProof> for Signature {
    fn from(from: SyncSelectionProof) -> Signature {
        from.0
    }
}

impl From<Signature> for SyncSelectionProof {
    fn from(sig: Signature) -> Self {
        Self(sig)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{FixedBytesExtended, MainnetEthSpec};
    use eth2_interop_keypairs::keypair;

    #[test]
    fn proof_sign_and_verify() {
        let slot = Slot::new(1000);
        let subcommittee_index = 12;
        let key = keypair(1);
        let fork = &Fork::default();
        let genesis_validators_root = Hash256::zero();
        let spec = &ChainSpec::mainnet();

        let proof = SyncSelectionProof::new::<MainnetEthSpec>(
            slot,
            subcommittee_index,
            &key.sk,
            fork,
            genesis_validators_root,
            spec,
        );
        assert!(proof.verify::<MainnetEthSpec>(
            slot,
            subcommittee_index,
            &key.pk,
            fork,
            genesis_validators_root,
            spec
        ));
    }
}
