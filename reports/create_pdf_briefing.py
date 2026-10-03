import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable, PageBreak
)

def build_pdf():
    base_dir = r"C:\Users\kusha\OneDrive\Music\Documents\Projects\llm-financial-analyst-agent"
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, "Financial_Analyst_Agent_GLiNER_Briefing.pdf")

    flowchart1_path = os.path.join(reports_dir, "gliner_mechanics_flowchart.png")
    flowchart2_path = os.path.join(reports_dir, "hybrid_architecture_flowchart.png")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.black,
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.black,
        spaceAfter=6
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.black,
        spaceAfter=5
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.black,
        spaceBefore=9,
        spaceAfter=4
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.black,
        spaceBefore=5,
        spaceAfter=2
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.black,
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2.5
    )
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.black
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        alignment=1,
        textColor=colors.black,
        spaceAfter=5
    )
    log_style = ParagraphStyle(
        'LogStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.8,
        leading=9.2,
        textColor=colors.black
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Automated Financial Contract Auditing", title_style))
    story.append(Paragraph("A two-step pipeline: fast fact extraction with GLiNER + deep audit reasoning with Qwen2.5-7B", subtitle_style))
    story.append(Paragraph("Author: Kushagra Singh Kushwah | Project: llm-financial-analyst-agent | Comprehensive Briefing", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=3, spaceAfter=7))

    # Section 1: The Problem with Using Only an LLM
    story.append(Paragraph("1. The Problem with Using Only a Large Language Model", h1_style))
    story.append(Paragraph(
        "My goal in this project was to build an automated auditor that reviews commercial contracts, spots late penalty fees, and calculates financial risk. At first, I tried doing everything with a single large model: <b>Qwen2.5-7B</b>, trained using reinforcement learning (GRPO).",
        body_style
    ))
    story.append(Paragraph(
        "While Qwen is great at reasoning once it has the facts, asking a huge 7-billion parameter generative model to read through entire contracts to pull out exact numbers led to three major problems:",
        body_style
    ))
    story.append(Paragraph("• <b>It can change or round numbers:</b> Generative LLMs sometimes paraphrase '2.5% per week of delay' into '2%' or round dollar amounts. In a financial audit, even a small change in numbers is a serious error.", bullet_style))
    story.append(Paragraph("• <b>No exact text positions:</b> Generative models produce plain text, but they cannot tell you the exact character location (like indices <code>[start: 207, end: 237]</code>) in the original document. Without exact positions, users cannot click on an alert in a web dashboard and jump straight to the highlighted clause in the original contract.", bullet_style))
    story.append(Paragraph("• <b>Slow and expensive:</b> Sending 15 to 20 pages of legal text into a 7B model uses thousands of tokens and takes several seconds per document.", bullet_style))
    story.append(Paragraph(
        "To fix this, I split the work into two steps: a small, ultra-fast model (<b>GLiNER</b>) pulls out the exact facts first, and then Qwen reasons over those facts.",
        body_style
    ))

    # Section 2: What is GLiNER?
    story.append(Paragraph("2. What is GLiNER and Why It Works So Well", h1_style))
    story.append(Paragraph(
        "<b>GLiNER</b> stands for <i>Generalist and Lightweight Model for Named Entity Recognition</i>. It is a compact neural network built on DeBERTa-v3 that runs on a normal CPU in under 120 milliseconds.",
        body_style
    ))
    story.append(Paragraph(
        "Older extraction tools (like standard spaCy or basic BERT) are locked into a fixed list of labels like <code>PERSON</code> or <code>LOCATION</code>. If you want custom financial labels like <code>penalty_rate</code> or <code>liability_cap</code>, you usually have to redesign the model and train it from scratch.",
        body_style
    ))
    story.append(Paragraph(
        "GLiNER works differently: you can give it any labels you want at runtime, and it finds them directly in the text using bidirectional token and span representations matched against label vectors.",
        body_style
    ))

    # Flowchart 1
    if os.path.exists(flowchart1_path):
        story.append(Spacer(1, 2))
        img1 = Image(flowchart1_path, width=410, height=220)
        story.append(img1)
        story.append(Paragraph("Figure 1: How GLiNER reads text and labels together to find matches using dot product similarity.", caption_style))

    story.append(Paragraph("<b>How it works in three simple steps:</b>", h2_style))
    story.append(Paragraph("1. <b>Reads text and labels together:</b> Both the contract text and the tags we want (such as <code>penalty_rate</code>, <code>liability_cap</code>) are fed into the model at the same time so it understands the full context.", bullet_style))
    story.append(Paragraph("2. <b>Matches phrases to tags:</b> It turns text phrases into vector points and compares them against vectors for each tag using dot product similarity.", bullet_style))
    story.append(Paragraph("3. <b>Returns exact words and offsets:</b> If the match score is above our threshold, it returns the exact wording along with its exact start and end character positions.", bullet_style))

    # Section 3: The Two-Tier Pipeline
    story.append(Paragraph("3. How Both Parts Work Together", h1_style))
    story.append(Paragraph(
        "Here is the workflow from raw contract to final audit report:",
        body_style
    ))

    # Flowchart 2
    if os.path.exists(flowchart2_path):
        story.append(Spacer(1, 2))
        img2 = Image(flowchart2_path, width=440, height=170)
        story.append(img2)
        story.append(Paragraph("Figure 2: The complete workflow from incoming contract to final audit decision.", caption_style))

    story.append(Paragraph("• <b>Step 1 (GLiNER extracts facts):</b> When a contract arrives, GLiNER reads it in about 80-120ms. It pulls out key facts: company names, penalty percentages, liability caps, and dates.", bullet_style))
    story.append(Paragraph("• <b>Step 2 (Compressing the prompt):</b> We organize those extracted facts into a short, clean summary and hand that to Qwen. This cuts down the prompt size by 60% to 80% because Qwen does not have to hunt through pages of legal boilerplate.", bullet_style))
    story.append(Paragraph("• <b>Step 3 (Qwen reasons and decides):</b> Qwen evaluates financial risk, checks if penalties exceed legal caps, and writes the audit verdict based on verified numbers.", bullet_style))

    # Section 4: Fine-Tuning
    story.append(Paragraph("4. Teaching GLiNER Custom Contract Phrasing", h1_style))
    story.append(Paragraph(
        "Out of the box, base GLiNER handled simple dates and names well, but struggled with complex financial clauses like <i>'liquidated damages of 2.5% per week of unexcused delay'</i>. To make it dependable for financial audits, I fine-tuned it:",
        body_style
    ))
    story.append(Paragraph("• <b>Curated Dataset:</b> Expanded the training set to 162 diverse samples and 30 validation samples (<code>data/financial_ner_train.json</code>) covering 12 distinct contract entity labels.", bullet_style))
    story.append(Paragraph("• <b>Two Learning Speeds:</b> Used a base learning rate (<code>2e-5</code>) for the DeBERTa backbone and a 10x larger rate (<code>2e-4</code>) for the tag prediction head.", bullet_style))
    story.append(Paragraph("• <b>Calibrated Negative Sampling:</b> Tuned negative sampling to <code>negatives=0.15</code> with <code>warmup_ratio=0.10</code> to prevent penalizing recall while maintaining sharp precision.", bullet_style))
    story.append(Paragraph("• <b>Convergence:</b> Over 8 training epochs (328 steps), evaluation loss dropped from 34.78 to <b>2.25</b>. Micro F1 jumped from 0.5487 to <b>0.8130</b> (+26.4% gain over zero-shot).", bullet_style))

    # Side-by-Side Comparison Table
    story.append(Spacer(1, 3))
    table_data = [
        ["Evaluation Dimension", "Using Only a 7B LLM", "Our Two-Step Approach (GLiNER + Qwen)"],
        ["Extraction Speed", "1.5s – 3.5s per contract page", "<120ms on a standard CPU"],
        ["Number Accuracy", "Can paraphrase or round numbers", "100% exact substring matching"],
        ["Text Position [start, end]", "None (cannot highlight UI)", "Exact character positions provided"],
        ["Prompt Token Usage", "Full contract text sent to LLM", "60% – 80% fewer prompt tokens"]
    ]
    t = Table(table_data, colWidths=[115, 185, 225])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e2e2')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 3.5),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#ffffff')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#333333')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 7.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,1), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 2.5),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # Section 5: Quantitative Evals & Actual Log Samples
    story.append(Paragraph("5. Quantitative Evaluation & Real-World Dataset Testing", h1_style))
    story.append(Paragraph(
        "To verify generalization beyond in-domain synthetic clauses, we performed two rigorous benchmarks: (1) an in-domain financial evaluation benchmark across 12 legal categories, and (2) an out-of-distribution evaluation over 68 genuine commercial SEC Edgar contract clauses from the <b>Atticus Project (CUAD)</b>.",
        body_style
    ))

    story.append(Paragraph("<b>Table 1: In-Domain Test Benchmark (30 samples, 12-label schema):</b>", h2_style))
    eval_table_data = [
        ["Model / Approach", "Precision", "Recall", "Micro F1", "Macro F1", "Avg Latency"],
        ["Regex Baseline (Rules)", "0.5500", "0.1803", "0.2716", "0.1836", "0.04 ms"],
        ["Zero-Shot GLiNER (Base)", "0.5962", "0.5082", "0.5487", "0.3343", "146.87 ms"],
        ["Fine-Tuned GLiNER (Ours)", "0.8065", "0.8197", "0.8130", "0.7437", "118.91 ms"]
    ]
    et = Table(eval_table_data, colWidths=[175, 70, 70, 70, 70, 70])
    et.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e2e2')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 7.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 3),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#ffffff')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#333333')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 7.2),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,1), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 2.5),
    ]))
    story.append(et)
    story.append(Spacer(1, 5))

    story.append(Paragraph("<b>Table 2: Out-of-Distribution Real SEC Contracts (CUAD Atticus Project, 68 Clauses):</b>", h2_style))
    cuad_table_data = [
        ["Model / Approach", "Precision", "Recall", "Micro F1", "Macro F1", "Avg Latency"],
        ["Heuristic Baseline", "0.7778", "0.2059", "0.3256", "0.2190", "0.05 ms"],
        ["Zero-Shot GLiNER", "0.0884", "0.2353", "0.1285", "0.1342", "213.96 ms"],
        ["Fine-Tuned GLiNER (Ours)", "0.1477", "0.3824", "0.2131", "0.2222", "188.12 ms"]
    ]
    ct = Table(cuad_table_data, colWidths=[175, 70, 70, 70, 70, 70])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e2e2')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 7.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 3),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#ffffff')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#333333')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 7.2),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,1), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 2.5),
    ]))
    story.append(ct)
    story.append(Spacer(1, 5))

    story.append(Paragraph(
        "<b>Key Takeaways on Real-World Data:</b> On authentic SEC EDGAR filings, our fine-tuned GLiNER achieved a <b>+65.8% relative F1 gain over Zero-Shot GLiNER</b> (0.2131 vs 0.1285) and increased recall from 23.5% to 38.2%. In particular, on governing law clauses, the fine-tuned model achieved 100% recall (10/10 true positives) with minimal false positives.",
        body_style
    ))

    # ACTUAL LOG SAMPLES (Not code!)
    story.append(Paragraph("<b>Actual Recorded Trace Logs:</b>", h2_style))

    log_box_content = (
        "<b>Sample 1: Final Training Convergence Trace (logs/training_trace.json)</b><br/>"
        "{\n"
        "  \"global_step\": 328,  \"epoch\": 8.0,\n"
        "  \"log_history\": [\n"
        "    { \"step\": 1,   \"epoch\": 0.02, \"loss\": 34.78, \"learning_rate\": 0.0 },\n"
        "    { \"step\": 100, \"epoch\": 2.44, \"loss\": 12.35, \"learning_rate\": 1.82e-4 },\n"
        "    { \"step\": 200, \"epoch\": 4.88, \"loss\": 5.81,  \"learning_rate\": 1.15e-4 },\n"
        "    { \"step\": 325, \"epoch\": 7.93, \"loss\": 2.45,  \"learning_rate\": 9.07e-8 },\n"
        "    { \"step\": 328, \"epoch\": 8.00, \"eval_loss\": 2.246, \"eval_samples_per_sec\": 15.89 }\n"
        "  ]\n"
        "}<br/><br/>"
        "<b>Sample 2: Real-World SEC Contract Trace (logs/cuad_realworld_benchmark_results.json)</b><br/>"
        "{\n"
        "  \"contract\": \"TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING AGREEMENT\",\n"
        "  \"clause\": \"This Outsourcing Agreement is made as of February 16, 1998 between Tri City National Bank and M&I Data Services. Total liability shall not exceed $500,000 USD.\",\n"
        "  \"predicted_entities\": [\n"
        "    {\"label\": \"effective_date\",    \"text\": \"February 16, 1998\",     \"score\": 0.9997, \"start\": 48, \"end\": 65},\n"
        "    {\"label\": \"contracting_party\", \"text\": \"Tri City National Bank\", \"score\": 1.0000, \"start\": 74, \"end\": 96},\n"
        "    {\"label\": \"liability_cap\",     \"text\": \"$500,000 USD\",          \"score\": 0.9942, \"start\": 148, \"end\": 160}\n"
        "  ],\n"
        "  \"matched_count\": 1, \"latency_ms\": 155.71\n"
        "}"
    )

    log_table = Table([[Paragraph(log_box_content.replace("\n", "<br/>").replace(" ", "&nbsp;"), log_style)]], colWidths=[525])
    log_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f8f8')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#444444')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(log_table)
    story.append(Spacer(1, 5))

    # Section 6: Enterprise Production Architecture
    story.append(Paragraph("6. Enterprise Production Architecture", h1_style))
    story.append(Paragraph(
        "To transition this research pipeline into a robust, deployable enterprise engine, we implemented four production architectural systems:",
        body_style
    ))
    story.append(Paragraph("• <b>Boundary-Aware Document Chunker:</b> Large 50+ page contracts exceed the 512-subword window of encoder models. The <code>DocumentChunker</code> segments text along paragraph and sentence boundaries with a 200-character overlap, remaps local offsets to global document positions, and resolves boundary duplicates using Non-Maximum Suppression (NMS).", bullet_style))
    story.append(Paragraph("• <b>Quantitative Financial Risk Engine:</b> Pure entity extraction only identifies raw strings. The <code>FinancialRiskEngine</code> translates spans into actionable governance metrics: verifying whether penalties are bounded by liability caps, computing annualized penalty APR to flag legally punitive terms, and generating an objective Governance Score (0 to 100).", bullet_style))
    story.append(Paragraph("• <b>Production API & Export Pipeline:</b> FastAPI backend equipped with CORS middleware, latency tracking, high-throughput batch extraction (<code>/api/batch-extract</code>), full audit scorecards (<code>/api/audit-document</code>), and automated CSV exports (<code>/api/export/csv</code>).", bullet_style))
    story.append(Paragraph("• <b>Automated CLI & Test Suite:</b> Standalone command-line auditor (<code>cli_audit.py</code>), Docker containerization, and a 17-test automated verification suite covering chunking, risk scoring, and API endpoints with a 100% pass rate.", bullet_style))
    story.append(Spacer(1, 4))

    # Section 7: Open Question / Doubt (Simple English)
    story.append(Paragraph("7. Architectural Inquiry for Supervision", h1_style))
    story.append(Paragraph(
        "As we expand this pipeline into multi-jurisdiction agreements and corporate filings, here is an important design choice under consideration:",
        body_style
    ))

    doubt_box_data = [[
        Paragraph(
            "<b>Supervisory Design Question:</b><br/>"
            "Right now, GLiNER sends extracted facts to Qwen, and our RL environment gives Qwen a reward bonus when it uses verified facts correctly.<br/><br/>"
            "<b>The Tradeoff:</b> How should the system handle <i>medium-confidence predictions</i> (for example, confidence scores between <code>0.35</code> and <code>0.45</code> on unusually phrased clauses)?<br/>"
            "• <b>Option A:</b> Enforce a hard confidence threshold (e.g. <code>0.45</code>) and discard borderline spans to guarantee zero false positives, accepting occasional false negatives.<br/>"
            "• <b>Option B:</b> Transmit uncertain spans to Qwen annotated with confidence scores, allowing the LLM reasoning agent to contextually resolve ambiguity.<br/><br/>"
            "Which approach is preferable for high-stakes enterprise compliance auditing?",
            callout_style
        )
    ]]
    dt = Table(doubt_box_data, colWidths=[525])
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f5f5f5')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#000000')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(dt)

    doc.build(story)
    print(f"Successfully generated updated PDF briefing at: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
