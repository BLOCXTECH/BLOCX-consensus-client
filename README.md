
BLOCX: Consensus Client
===============================================

**BLOCX** is a customized Ethereum consensus client, originally forked from [Lighthouse](https://github.com/sigp/lighthouse) and modified to support the BLOCX blockchain network.

* * *

 Overview
-----------

BLOCX is:

*   A fork of the Ethereum 2.0 consensus client **Lighthouse**, tailored for the **BLOCX** chain.
    
*   Built in **Rust**, leveraging its memory safety and performance benefits.
    
*   Designed to support the **BLOCX** proof-of-stake network.
    
*   Open-source and licensed under **Apache 2.0**.
    
*   Maintained by the BLOCX community.
    

This fork includes protocol-level changes, configuration updates, and customized behavior unique to the BLOCX chain.

* * *

 What’s Different?
--------------------

Compared to Lighthouse:

*   Custom **chain configuration** and **genesis setup** for the BLOCX chain.
    
*   Modified **consensus rules** and **network parameters**.
    
*   Unique staking and validator logic tailored for BLOCX.
    
*   Adapted codebase to support **BLOCX-specific upgrades**.
    

* * *

 Documentation
----------------

Coming soon: [BLOCX Docs](https://blocx.gitbook.io/blocx.) – includes validator setup, network parameters, node deployment, and FAQs.

* * *

 Staking Deposit Contract
--------------------------

The BLOCX mainnet recognizes:

`0x00000000219ab540356cBB839Cbe05303d7705Fa`

as the official deposit contract for validators.

* * *

 How to Build
---------------

    git clone https://github.com/BLOCXTECH/blocx-consensus-client.git
    cd blocx-consensus-client
    make
    

* * *

 Credits
----------

Original work by [Sigma Prime](https://sigmaprime.io/) on [Lighthouse](https://github.com/sigp/lighthouse).  

* * *
