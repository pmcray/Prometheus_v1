import Lake
open Lake DSL

-- Prometheus v1 – Hidden-Law Lab Lean layer
-- Toolchain : leanprover/lean4:v4.34.1  (see lean-toolchain)
-- Mathlib   : pinned at bc70da8fd1d791a61ad9089b6a8da1dc671b9198
--             (inherited from Prometheus v0 PoC lake-manifest.json)
--
-- Ground-truth proofs use only Lean 4 core (Fin + decide) and do NOT
-- import Mathlib.  This project exists solely to record the Mathlib pin
-- for future use (e.g., Goedel-Prover-V2 which needs Mathlib tactics).

package PrometheusLaws where
  name := "PrometheusLaws"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "bc70da8fd1d791a61ad9089b6a8da1dc671b9198"

-- Ground-truth library: auto-generated, stdlib-only files
lean_lib PrometheusLaws where
  srcDir := "."
