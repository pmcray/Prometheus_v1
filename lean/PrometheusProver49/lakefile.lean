import Lake
open Lake DSL

-- Prometheus v1 — Phase 3 Prover Bake-Off Lean 4.9 Toolchain Project
-- Toolchain : leanprover/lean4:v4.9.0 (Goedel-Prover-V2-8B native target)
-- Mathlib   : v4.9.0 tag (optional; core Hidden-Law Lab theorems compile with
--             Lean 4.9 stdlib only, and Colab Cell 4d can fetch prebuilt Mathlib
--             olean cache with `lake exe cache get` if Mathlib tactics are used).
--
-- Kept strictly separate from `lean/PrometheusLaws/` (`v4.34.1`) so the 1,000-world
-- ground-truth `decide` verification certificate is never disturbed.

package PrometheusProver49 where
  name := "PrometheusProver49"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.9.0"

lean_lib PrometheusProver49 where
  srcDir := "."
