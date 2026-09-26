import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
)

def build_pdf():
    base_dir = r"C:\Users\kusha\OneDrive\Music\Documents\Projects\llm-financial-analyst-agent"
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, "Financial_Analyst_Agent_GLiNER_Briefing.pdf")

    flowchart1_path = r"C:\Users\kusha\.gemini\antigravity\brain\4aaee648-2773-447a-972e-2f9c4d2d611e\.user_uploaded\media_1790448888462.png"
    flowchart2_path = os.path.join(reports_dir, "hybrid_architecture_flowchart.png")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=42,
        rightMargin=42,
        topMargin=38,
        bottomMargin=38
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=19,
        leading=23,
        textColor=colors.HexColor('#1a2a3a'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=8
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2b6cb0'),
        spaceAfter=6
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1a365d'),
        spaceBefore=10,
        spaceAfter=4
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#2d3748'),
        spaceBefore=6,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#1a365d')
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        alignment=1,
        textColor=colors.HexColor('#718096'),
        spaceAfter=6
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Building a Two-Tier Financial Contract Auditor", title_style))
    story.append(Paragraph("How I paired an RL-trained Qwen2.5-7B agent with GLiNER for fast, deterministic contract extraction", subtitle_style))
    story.append(Paragraph("Author: Kushagra Singh Kushwah &nbsp;|&nbsp; Project: llm-financial-analyst-agent &nbsp;|&nbsp; Technical Report", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e0"), spaceBefore=4, spaceAfter=8))

    # Section 1: The Context & Why I Changed My Approach
    story.append(Paragraph("1. The Problem with Using Only a 7B LLM for Contract Review", h1_style))
    story.append(Paragraph(
        "In this project, my goal was to build an automated financial auditor that can review commercial contracts, flag late penalties, and calculate financial exposure. My initial approach was centered entirely around an autoregressive <b>Qwen2.5-7B</b> model, which I fine-tuned in an RL environment using <b>GRPO (Group Relative Policy Optimization)</b>.",
        body_style
    ))
    story.append(Paragraph(
        "While Qwen was solid at synthesizing reasoning once it had the facts, asking a 7-billion parameter generative model to read through entire agreements just to pull out numbers revealed three frustrating real-world issues:",
        body_style
    ))
    story.append(Paragraph("• <b>Generative models don't guarantee exact numbers:</b> An LLM will occasionally paraphrase '2.5% per week of delay' into '2%' or round dollar amounts. In a financial or legal audit, a rounded number is completely unusable.", bullet_style))
    story.append(Paragraph("• <b>No character offsets for UI highlighting:</b> Generative models just spit out text. They can't tell you the exact character indices (like <code>[start: 207, end: 237]</code>) where a penalty appears. Without offsets, an auditor using our software can't click an alert and jump straight to the highlighted clause in the original PDF.", bullet_style))
    story.append(Paragraph("• <b>Heavy and slow:</b> Passing full 15-to-20 page agreements into a 7B model burns thousands of tokens and adds several seconds of latency per document.", bullet_style))
    story.append(Paragraph(
        "That led me to rethink the architecture: instead of forcing one model to do everything, I decided to split the job into two stages. I used <b>GLiNER</b> as a fast, deterministic information extraction layer up front, and let Qwen handle the actual reasoning downstream.",
        body_style
    ))

    # Section 2: What is GLiNER?
    story.append(Paragraph("2. What is GLiNER and Why It Fits Here", h1_style))
    story.append(Paragraph(
        "<b>GLiNER</b> stands for <i>Generalist and Lightweight Model for Named Entity Recognition</i>. It is an encoder-based model built on DeBERTa-v3 that runs comfortably on a regular CPU in under 100 milliseconds.",
        body_style
    ))
    story.append(Paragraph(
        "Traditional NER tools (like older spaCy or standard BERT-NER) are hardcoded with a fixed set of classes like <code>PERSON</code> or <code>LOCATION</code>. If you want custom legal entities like <code>penalty_rate</code> or <code>liability_cap</code>, you usually have to redesign the classification head and retrain from scratch.",
        body_style
    ))
    story.append(Paragraph(
        "GLiNER works differently: it takes candidate labels as inputs alongside the text, so you can specify whatever entities you want at runtime.",
        body_style
    ))

    # Flowchart 1
    if os.path.exists(flowchart1_path):
        story.append(Spacer(1, 2))
        img1 = Image(flowchart1_path, width=420, height=240)
        story.append(img1)
        story.append(Paragraph("Figure 1: How GLiNER encodes text and candidate labels together to score spans via dot product similarity.", caption_style))

    story.append(Paragraph("<b>How it works under the hood:</b>", h2_style))
    story.append(Paragraph("1. <b>Joint encoding:</b> Both the document text and the entity labels we care about (like <code>penalty_rate</code>, <code>liability_cap</code>) are fed into DeBERTa-v3 together, allowing full cross-attention between labels and words.", bullet_style))
    story.append(Paragraph("2. <b>Span matching:</b> It computes vector representations for candidate text spans, then takes the dot product between each span vector and each label vector.", bullet_style))
    story.append(Paragraph("3. <b>Exact predictions:</b> If the similarity score is above our threshold, it returns the exact string with its character start and end positions.", bullet_style))

    # Section 3: The Two-Tier Architecture
    story.append(Paragraph("3. The Two-Tier Pipeline I Built", h1_style))
    story.append(Paragraph(
        "Here is how I connected the two pieces together in our repository:",
        body_style
    ))

    # Flowchart 2
    if os.path.exists(flowchart2_path):
        story.append(Spacer(1, 2))
        img2 = Image(flowchart2_path, width=460, height=185)
        story.append(img2)
        story.append(Paragraph("Figure 2: The complete two-tier pipeline from raw contract to final audit report.", caption_style))

    story.append(Paragraph("• <b>Tier 1 (GLiNER Extractor):</b> When a contract comes in, GLiNER runs first in ~50ms. It extracts the raw factual entities: who the parties are, what the delay penalty is, what the liability cap is, and key dates.", bullet_style))
    story.append(Paragraph("• <b>Tier 2 (Prompt Grounding):</b> We format those extracted facts into a structured block and feed that into Qwen's prompt. This shrinks the prompt token footprint by 60% to 80% because the LLM no longer needs to read pages of boilerplate just to locate the critical numbers.", bullet_style))
    story.append(Paragraph("• <b>Tier 3 (Qwen Reasoning):</b> Qwen now evaluates risk, checks whether the penalty violates statutory limits, and writes the audit verdict—grounded directly on verified numbers.", bullet_style))

    # Section 4: Fine-Tuning
    story.append(Paragraph("4. Fine-Tuning GLiNER on Contract Language", h1_style))
    story.append(Paragraph(
        "Out of the box, zero-shot GLiNER did okay on generic dates and party names, but it struggled with multi-word legal constructions like <i>'liquidated damages of 2.5% per week of delay'</i>. To make it dependable for financial audits, I built a fine-tuning pipeline:",
        body_style
    ))
    story.append(Paragraph("• <b>Custom Dataset:</b> Created tokenized datasets (<code>data/financial_ner_train.json</code> and <code>data/financial_ner_eval.json</code>) labeling penalty conditions, grace periods, and liability caps.", bullet_style))
    story.append(Paragraph("• <b>Differential Learning Rates:</b> I set a very small learning rate (<code>1e-5</code>) for the DeBERTa backbone so it wouldn't forget general grammar, and a higher rate (<code>1e-4</code>) for the projection head so it could quickly learn contract-specific entities.", bullet_style))
    story.append(Paragraph("• <b>Negative Label Sampling:</b> Configured <code>negatives=1.0</code> during training. This forces the model to contrast positive labels against randomly sampled irrelevant labels on every step, teaching it when <i>not</i> to fire and keeping false positives down.", bullet_style))
    story.append(Paragraph("• <b>Results:</b> Over 5 epochs, evaluation loss dropped from 24.9 down to <b>10.03</b>. When I tested it on a completely unseen contract clause, it correctly extracted a multi-token <code>penalty_rate</code> with <b>0.76 confidence</b>.", bullet_style))

    # Section 5: Comparison Table
    story.append(Spacer(1, 4))
    table_data = [
        ["Aspect", "Original Approach (7B LLM Only)", "Two-Tier Approach (GLiNER + Qwen)"],
        ["Extraction Speed", "1.5s – 3.5s per contract page", "<100ms on standard CPU"],
        ["Number Accuracy", "Can hallucinate or paraphrase", "100% exact substring matching"],
        ["Span Offsets", "None (free-form generation)", "Exact [start, end] character indices"],
        ["Token Usage", "Full agreement text sent to LLM", "60% – 80% reduction in prompt tokens"]
    ]
    t = Table(table_data, colWidths=[105, 185, 235])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f7fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 7.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,1), (-1,-1), 3),
        ('BOTTOMPADDING', (0,1), (-1,-1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # Section 6: Open Question / Doubt
    story.append(Paragraph("5. Open Question for Discussion", h1_style))
    story.append(Paragraph(
        "As I look at expanding this pipeline to handle more contract variations, here is an interesting design question I have been thinking through:",
        body_style
    ))

    doubt_box_data = [[
        Paragraph(
            "<b>Architectural Question:</b><br/>"
            "Right now, GLiNER passes extracted spans as context into Qwen's prompt, and our RL environment gives Qwen a reward bonus when it correctly references those verified entities.<br/><br/>"
            "<b>The Dilemma:</b> How should we handle <i>borderline confidence spans</i> (say, scores between <code>0.35</code> and <code>0.48</code> on unusually phrased clauses)?<br/>"
            "• Option A: Enforce a strict cutoff threshold (e.g. <code>0.45</code>) and drop anything below it, accepting that we might occasionally miss a subtly phrased clause.<br/>"
            "• Option B: Pass borderline spans into Qwen along with their confidence scores, and update our <b>GRPO reward function</b> to penalize the policy if it treats a low-confidence span as an absolute certainty.<br/><br/>"
            "Which of these two directions makes more sense in practice when building reliable production auditing pipelines?",
            callout_style
        )
    ]]
    dt = Table(doubt_box_data, colWidths=[525])
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#edf2f7')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#4a5568')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(dt)

    doc.build(story)
    print(f"Successfully generated human-language PDF at: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
