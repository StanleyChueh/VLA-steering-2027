# S2d: is the source/training behaviour a privileged steering reference?

Label: **ORACLE DIAGNOSTIC.** Numbers are from `S2D_ORACLE_REFERENCE_RESULTS.md` /
`results/S2d/s2d_summary.json`.

## Design recap

* SOURCE-NORM and OTHER are both norm-matched to MASK at every policy call: same ‖ΔA‖ = ‖0.5 (A_f − A_u)‖.
* They differ only in which benchmark instruction defines the direction:
  * SOURCE: the training instruction of the identical LIBERO-Spatial scene. Its subject is the biased object.
  * OTHER: the lexicographically first remaining instruction. In G0 that is the *other* black bowl (tasks 1, 4,
    5, 10) or the cookie box (task 11).
* SOURCE-RAW uses CAG's unnormalised formula with the source branch. Its median intervention norm is 26 % larger
  than MASK's.

## 1. In G0, the source reference is not special for producing the instructed behaviour

| G0 (100 pairs) | MASK | SOURCE-NORM | OTHER | SOURCE-RAW |
|---|---|---|---|---|
| faithful success | 0.02 | 0.06 | **0.07** | 0.08 |
| faithful first-contact | 0.02 | 0.08 | 0.07 | 0.11 |
| paired vs MASK (success gain/loss) | — | 4 / 0 | 5 / 0 | 7 / 1 |

* SN − OT: faithful success −0.01 (2/3), faithful first-contact +0.01 (3/2), p = 1.0 for both.
* Any norm-matched semantic contrast lifts G0 faithful outcomes from about 2 % to 6–8 %. Even that is not
  significant against MASK at the preregistered level.
* The actual source behaviour gives no extra benefit for reaching the instructed object.

## 2. The source reference is special for suppressing the source behaviour

| G0 | MASK | SOURCE-NORM | OTHER |
|---|---|---|---|
| biased success | 0.76 | **0.59** | 0.75 |
| biased first-contact | 0.86 | 0.75 | 0.81 |
| timeout | 0.16 | **0.29** | 0.12 |
| episodes ending with no labelled object touched (MASK biased-first → SN neither) | — | 8 | 3 (→ OT) |

Paired comparisons for biased success:
* SN − OT: −0.16 (2/18, p = 0.001).
* SN − MASK: −0.17 (5/22, p = 0.007).

Per task, the drop is in tasks 4 (−5 vs MASK), 5 (−3), 10 (−7) and 11 (−2). Seed-consistent losses in 5 states.

So pointing the guidance *away from the retrieved training trajectory* specifically and reproducibly suppresses
that trajectory, and OTHER does not. But the suppressed probability mass goes mostly to stalling and timeouts, not
to the instructed object. **The source reference is a privileged suppressor, not a redirector.** Removing the
competing mode does not reveal a correct one.

## 3. Direction vs magnitude (SOURCE-RAW vs SOURCE-NORM, secondary)

* SR − SN in G0:
  * faithful success +0.02 (5/3, p = 0.74);
  * faithful first-contact +0.03 (4/1, p = 0.50);
  * biased success +0.07 (10/3, p = 0.12).
* The larger, unnormalised source push is not reliably better. Its extra first-contact gains are all in task 4
  (+3), and it suppresses the biased behaviour *less* than SN.
* On the control tasks it is the worst condition for execution:
  * post-contact failure 0.31;
  * faithful success 0.51 vs MASK 0.63 (p = 0.026).
* There is no sign that the G0 failure is an amplitude problem that a bigger source push would solve. This agrees
  with S2c-0's descriptive finding that zero-rescue tasks are not amplitude-limited.

## 4. Control tasks: reference choice moves grounding and execution in opposite directions

| control (160 pairs) | MASK | SN | OT | SR |
|---|---|---|---|---|
| faithful first-contact | 0.73 | 0.79 | 0.76 | 0.79 |
| faithful success | 0.63 | 0.58 | 0.59 | 0.51 |
| post-contact failure | 0.12 | 0.25 | 0.20 | 0.31 |

* Source-referenced guidance reaches the right object somewhat more often than masked CAG (SN − MASK first-contact
  +0.069, p = 0.079).
* It then fails after contact about twice as often. The net task success is slightly lower: −5.6 pp, p = 0.24.
* Task-level, the success losses concentrate in tasks 2 (−6) and 7 (−4). Both are cookie-box tasks; in task 7
  the source bowl sits next to the cookie box, in task 2 next to the ramekin.
* This is the S2b "execution-limited" pattern again: a steering direction that helps grounding interferes with
  manipulation after contact. The effect is larger for the source reference than for the language-masked one.

## 5. Specificity verdict

| question | answer |
|---|---|
| Does the oracle source reference unlock the zero-rescue regime? | **No.** Faithful success 6 % (vs MASK 2 %, p = 0.13); task 1 never moves |
| Is the source reference better than an arbitrary benchmark alternative at producing faithful behaviour? | **No.** SN ≈ OT (6 % vs 7 %) |
| Is it better at suppressing the retrieved source behaviour? | **Yes.** Biased success 59 % vs OT 75 % (p = 0.001) and vs MASK 76 % (p = 0.007), mostly into timeouts |
| Is a bigger push the missing ingredient? | **No evidence.** SOURCE-RAW ≈ SOURCE-NORM in G0, and worse on control execution |

Taken together: in the zero-rescue tasks, the frozen π0.5 appears to lack an accessible instructed-object mode
under these prompts at call time. Knowing exactly which competing trajectory it retrieved lets steering push away
from that trajectory, but not toward the instructed object. This is evidence against "reference choice is the
bottleneck". It is weakly in favour of the instruction-action *binding* failure being upstream of any action-space
steering. That last reading is an interpretation and is not tested here.
