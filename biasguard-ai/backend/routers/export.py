from fastapi import APIRouter, HTTPException, Response
from models.schemas import AuditExportRequest
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from services.groq_advisor import GroqAdvisor

router = APIRouter(prefix="/api/export")
advisor = GroqAdvisor()

@router.post("/json")
async def export_json(request: AuditExportRequest):
    try:
        # The request is already structured according to AuditExportRequest
        report_data = request.dict()
        return report_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON export failed: {str(e)}")

@router.post("/pdf")
async def export_pdf(request: AuditExportRequest):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1, # Center
            textColor=colors.hexColor("#00f5d4")
        )
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.hexColor("#00f5d4")
        )
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=styles['Normal'],
            fontSize=8,
            alignment=1,
            textColor=colors.gray
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            textColor=colors.black
        )

        elements = []

        # Cover Page
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("BiasGuard AI — Fairness Audit Report", title_style))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f"Dataset: {request.metadata.filename}", styles['Normal']))
        elements.append(Paragraph(f"Generated on: {request.timestamp}", styles['Normal']))
        elements.append(Spacer(1, 1*inch))
        elements.append(Paragraph("Confidential Audit Result", styles['Italic']))
        elements.append(PageBreak())

        # Executive Summary page (Groq narrative, optional and non-blocking)
        if request.include_groq_narrative and advisor.is_configured():
            try:
                payload = request.dict()
                narrative = await advisor.generate_audit_report_narrative(payload)
                if narrative:
                    elements.append(Paragraph("Executive Summary (AI Narrative)", section_style))
                    for para in [p.strip() for p in narrative.split("\n") if p.strip()]:
                        elements.append(Paragraph(para, body_style))
                        elements.append(Spacer(1, 0.15 * inch))
                    elements.append(PageBreak())
            except Exception:
                # Never block export when Groq is unavailable.
                pass

        # Section 1: Bias Summary
        elements.append(Paragraph("Section 1: Bias Summary", section_style))
        di = request.baseline_metrics.fairness_metrics.demographic_parity_difference
        headline = "Bias Detected" if abs(di) > 0.1 else "Low Bias Detected"
        elements.append(Paragraph(f"Headline: {headline}", styles['Heading3']))
        
        metrics_data = [
            ["Metric", "Value"],
            ["Accuracy", f"{request.baseline_metrics.accuracy:.4f}"],
            ["Demographic Parity Diff", f"{di:.4f}"],
            ["Disparate Impact Ratio", f"{request.baseline_metrics.fairness_metrics.disparate_impact_ratio:.4f}"]
        ]
        t1 = Table(metrics_data, colWidths=[3*inch, 2*inch])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.hexColor("#0f0f19")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.hexColor("#f9f9f9")),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t1)
        elements.append(Spacer(1, 0.2*inch))

        # Section 2: Strategy Comparison
        if request.simulation_results:
            elements.append(Paragraph("Section 2: Strategy Comparison", section_style))
            strat_data = [["Strategy", "Fairness Gain", "Accuracy Drop"]]
            for s in request.simulation_results:
                strat_data.append([s.strategy_name, f"+{s.fairness_gain*100:.1f}%", f"-{s.accuracy_drop*100:.1f}%"])
            
            t2 = Table(strat_data, colWidths=[2.5*inch, 1.25*inch, 1.25*inch])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.hexColor("#0f0f19")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t2)

        # Section 3: Recommended Strategy
        if request.recommendation:
            elements.append(Paragraph("Section 3: Recommended Strategy", section_style))
            rec = request.recommendation
            elements.append(Paragraph(f"Recommended: <b>{rec.recommended_strategy}</b>", styles['Normal']))
            elements.append(Paragraph(f"Reason: {rec.reason}", styles['Normal']))
            elements.append(Paragraph(f"Projected Fairness Improvement: +{rec.expected_fairness_gain*100:.1f}%", styles['Normal']))
            elements.append(Paragraph(f"Projected Accuracy Cost: -{rec.expected_accuracy_drop*100:.1f}%", styles['Normal']))

        # Section 4: Counterfactual Examples
        if request.counterfactual_examples:
            elements.append(Paragraph("Section 4: Counterfactual Examples (Sample)", section_style))
            cf_data = [["Sensitive Attribute", "Original Outcome", "Flipped Outcome"]]
            for cf in request.counterfactual_examples[:3]: # Up to 3
                cf_data.append([
                    f"{cf.sensitive_attr_original} -> {cf.sensitive_attr_flipped}",
                    "Rejected",
                    "Approved"
                ])
            
            t3 = Table(cf_data, colWidths=[2.5*inch, 1.25*inch, 1.25*inch])
            t3.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.hexColor("#0f0f19")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t3)

        # Footer
        elements.append(Spacer(1, 1*inch))
        elements.append(Paragraph("Generated by BiasGuard AI — Accuracy without Fairness is an Error", footer_style))

        doc.build(elements)
        pdf_content = buffer.getvalue()
        buffer.close()

        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=BiasGuard_Audit_{request.metadata.filename}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")
