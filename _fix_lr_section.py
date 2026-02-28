import json

with open('assignments/assignment_1_Team5_final.ipynb') as f:
    nb = json.load(f)

lines = nb['cells'][31]['source']

new_lr_section = [
    '#### Effect of Learning Rate\n',
    '\n',
    'The loss curves for the moderate learning rate experiments are shown in [Figure 1](#fig-lr-average), and the extreme learning rate experiments in [Figure 2](#fig-lr-extreme).\n',
    '\n',
    '<a id="fig-lr-average"></a>\n',
    '\n',
    '![Figure 1: Effect of Learning Rate, average experiments](https://raw.githubusercontent.com/helloimdomas/team5-medical-ai/main/assignments/task1deliverables/ex2_experiments-average_learning_rate.png)\n',
    '\n',
    '*Figure 1: Loss curves for moderate learning rate variations (lr=1e-4, baseline 1e-3, lr=1e-2).*\n',
    '\n',
    '<a id="fig-lr-extreme"></a>\n',
    '\n',
    '![Figure 2: Effect of Learning Rate, extreme experiments](https://raw.githubusercontent.com/helloimdomas/team5-medical-ai/main/assignments/task1deliverables/ex2_experiments-extreme_learning_rate.png)\n',
    '\n',
    '*Figure 2: Loss curves for extreme learning rate variations (lr=1e-5, baseline 1e-3, lr=5e-2).*\n',
    '\n',
    "- Low learning rate (1e-4): The optimizer takes very small steps, so the model learns slowly. After 2000 iterations the loss is still high (val=1.46 vs. baseline 1.12), indicating the model has not yet converged. Given more iterations, it would likely eventually reach a similar loss, but within this fixed budget it appears undertrained. This is not underfitting in the classical sense, the model has sufficient capacity, it simply hasn't had enough effective optimization steps.\n",
    '\n',
    '- High learning rate (1e-2): The optimizer takes overly large steps, causing it to overshoot minima. The loss curve shows instability because the loss occasionally rises before descending again (e.g. from 2.30 at iter 1400 back up to 2.33 at iter 1600), as visible in [Figure 1](#fig-lr-average). The final loss (val=2.04) is worse than even the low learning rate, because the large steps prevent the optimizer from settling into a good region of the loss landscape. This is optimization instability, not underfitting.\n',
    '\n',
    'The extreme parameter experiments confirm these patterns more dramatically (see [Figure 2](#fig-lr-extreme)):\n',
    '- lr=1e-5: Severe undertraining, the loss barely decreases from its initial value (val=2.71), because the tiny step size means the model has made almost no meaningful progress in 2000 iterations.\n',
    '- lr=5e-2: Clear divergence. The loss oscillates heavily throughout training (e.g. dropping to 2.71 at iter 1600, then rising back up to 3.04 at iter 2000). The optimizer never converges, it repeatedly overshoots the minima and the loss actually increases toward the end.\n',
    '\n',
    'Conclusion:\n',
    "The baseline learning rate of 1e-3 is well tuned for this model and dataset. Too low learning rates cause slow convergence (the model can't make enough progress within the iteration budget), while too high learning rates cause optimization instability or divergence (the optimizer overshoots and oscillates). The optimal learning rate depends on the specific interaction between the optimizer (AdamW maintains moving averages of gradients), the loss landscape shape, and the gradient magnitudes so it cannot be derived analytically and must be found empirically.\n",
    '\n',
    "To further demonstrate that the learning rate primarily affects the speed of convergence, we extended the training to 6000 iterations (3x the default budget) for both the baseline (`lr=1e-3`) and the low learning rate (`lr=1e-4`). With 6000 iterations, `lr=1e-4` reached train=0.9916 and val=1.0926, lower than the baseline at 2000 iterations (train~1.05, val~1.12), confirming that the model was indeed just undertrained and continued to improve given more time. However, the baseline `lr=1e-3` at 6000 iterations reached train=0.8085 and val=0.9532, which is substantially lower still. This shows that even with 3x more iterations, the lower learning rate cannot catch up to the default, and the optimizer's small step size remains a bottleneck, leaving the model underfitted relative to what the same architecture can achieve with a better tuned learning rate.\n",
    '\n',
    '<a id="fig-lr-6000"></a>\n',
    '\n',
    '![Figure 3: Effect of Learning Rate, 6000 iterations](https://raw.githubusercontent.com/helloimdomas/team5-medical-ai/main/assignments/task1deliverables/6000iters.png)\n',
    '\n',
    '*Figure 3: Loss curves for baseline (lr=1e-3) vs. low learning rate (lr=1e-4) over 6000 iterations.*\n',
    '\n',
    '\n',
]

# Replace lines 54-87 (the entire LR section up to "#### Effect of Batch Size")
lines[54:88] = new_lr_section

# Fix figure references in the Overfitting/Underfitting section
# Figure 2 (old average) -> Figure 1, Figure 3 (old extreme) -> Figure 2
for i in range(len(lines)):
    if i > 100 and ('fig-lr-average' in lines[i] or 'fig-lr-extreme' in lines[i]):
        lines[i] = lines[i].replace('[Figure 2](#fig-lr-average)', '[Figure 1](#fig-lr-average)').replace('[Figure 3](#fig-lr-extreme)', '[Figure 2](#fig-lr-extreme)')

nb['cells'][31]['source'] = lines

with open('assignments/assignment_1_Team5_final.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! Verifying...")
for i in range(54, 54 + len(new_lr_section)):
    print(f"{i:3d}: {lines[i][:100].rstrip()}")

print("\nOverfitting section references:")
for i in range(len(lines)):
    if i > 100 and 'fig-lr' in lines[i]:
        print(f"{i:3d}: {lines[i][:120].rstrip()}")
