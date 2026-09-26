import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up figure
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)
ax.axis('off')

# Neutral grayscale colors (zero blue tint)
box_border = "#000000"
box_fill1 = "#f8f9fa"
box_fill2 = "#f1f3f5"
box_fill3 = "#e9ecef"
box_fill4 = "#dee2e6"
box_fill5 = "#f8f9fa"
text_color = "#000000"
subtext_color = "#222222"

fig.patch.set_facecolor('white')

# Boxes definition
boxes = [
    {"x": 0.3, "y": 1.7, "w": 1.6, "h": 1.4, "title": "Raw Input\nContract", "desc": "PDF / Agreement\nClauses", "color": box_fill1},
    {"x": 2.2, "y": 1.2, "w": 2.1, "h": 2.4, "title": "Tier 1: GLiNER\nExtractor", "desc": "DeBERTa-v3 (<100ms CPU)\nExact Spans [start:end]:\n• penalty_rate: '2.5%'\n• liability_cap: '$750k'\n• contracting_party", "color": box_fill2},
    {"x": 4.6, "y": 1.7, "w": 1.6, "h": 1.4, "title": "Tier 2: Prompt\nShield", "desc": "Context Grounding\n& Factual Bounds\n(60-80% token cut)", "color": box_fill3},
    {"x": 6.5, "y": 1.5, "w": 1.7, "h": 1.8, "title": "Tier 3: Qwen2.5-7B\nGRPO Agent", "desc": "Reinforcement\nLearned Policy\nReasoning & Audit", "color": box_fill4},
    {"x": 8.5, "y": 1.7, "w": 1.3, "h": 1.4, "title": "Audit\nVerdict", "desc": "Risk Scores &\nCompliance\nReport", "color": box_fill5},
]

for b in boxes:
    rect = patches.FancyBboxPatch(
        (b["x"], b["y"]), b["w"], b["h"],
        boxstyle="round,pad=0.1,rounding_size=0.15",
        edgecolor=box_border, facecolor=b["color"], linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(b["x"] + b["w"]/2, b["y"] + b["h"] - 0.35, b["title"],
            ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_color)
    ax.text(b["x"] + b["w"]/2, b["y"] + (b["h"] - 0.5)/2, b["desc"],
            ha='center', va='center', fontsize=7.5, color=subtext_color)

# Arrows in solid black
arrows = [(1.9, 2.4, 2.2, 2.4), (4.3, 2.4, 4.6, 2.4), (6.2, 2.4, 6.5, 2.4), (8.2, 2.4, 8.5, 2.4)]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=box_border, edgecolor=box_border, arrowstyle='->', lw=1.8))

# Subtitle and titles in pure black
ax.text(5.0, 4.1, "Two-Tier Hybrid Financial Contract Auditing Pipeline", ha='center', fontsize=12, fontweight='bold', color=text_color)
ax.text(5.0, 3.8, "Fast Deterministic Information Extraction (GLiNER) + Deep Policy Reasoning (Qwen2.5-7B GRPO)", ha='center', fontsize=8.5, color="#333333")

plt.tight_layout()
output_img = r"C:\Users\kusha\OneDrive\Music\Documents\Projects\llm-financial-analyst-agent\reports\hybrid_architecture_flowchart.png"
import os
os.makedirs(os.path.dirname(output_img), exist_ok=True)
plt.savefig(output_img, bbox_inches='tight', dpi=300)
print(f"Saved neutral-color flowchart to {output_img}")
