"""
Report Generator for creating PDF and CSV reports of analysis results
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, List
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import io
import base64

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        
    async def generate_pdf_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a comprehensive PDF report"""
        try:
            task_id = analysis_result.get('task_id', 'unknown')
            report_filename = f"deepfake_analysis_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(report_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.darkblue,
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph("Digital Deepfake Detection Report", title_style))
            story.append(Spacer(1, 20))
            
            # Report metadata
            metadata_style = styles['Normal']
            story.append(Paragraph(f"<b>Analysis ID:</b> {task_id}", metadata_style))
            story.append(Paragraph(f"<b>Analysis Type:</b> {analysis_result.get('analysis_type', 'Unknown').title()}", metadata_style))
            story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", metadata_style))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            authenticity_score = analysis_result.get('authenticity_score', 0)
            confidence_level = analysis_result.get('confidence_level', 'Unknown')
            is_deepfake = analysis_result.get('detection_results', {}).get('is_deepfake', False)
            risk_level = analysis_result.get('detection_results', {}).get('risk_level', 'Unknown')
            
            summary_data = [
                ['Metric', 'Value', 'Assessment'],
                ['Authenticity Score', f"{authenticity_score}%", self._get_score_assessment(authenticity_score)],
                ['Confidence Level', confidence_level, ''],
                ['Deepfake Detection', 'Yes' if is_deepfake else 'No', ''],
                ['Risk Level', risk_level, '']
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Detailed Analysis
            story.append(Paragraph("Detailed Analysis", styles['Heading2']))
            
            detailed_analysis = analysis_result.get('detailed_analysis', {})
            if detailed_analysis:
                for category, data in detailed_analysis.items():
                    if isinstance(data, dict):
                        story.append(Paragraph(f"{category.replace('_', ' ').title()}", styles['Heading3']))
                        
                        # Create table for category data
                        category_data = [['Parameter', 'Value']]
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                if 0 <= value <= 1:
                                    value_str = f"{value:.3f}"
                                else:
                                    value_str = f"{value:.2f}"
                            elif isinstance(value, list):
                                value_str = f"{len(value)} items"
                            else:
                                value_str = str(value)[:50]  # Truncate long strings
                            
                            category_data.append([key.replace('_', ' ').title(), value_str])
                        
                        if len(category_data) > 1:  # Only add table if there's data
                            category_table = Table(category_data, colWidths=[3*inch, 2*inch])
                            category_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 10),
                                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                            ]))
                            story.append(category_table)
                            story.append(Spacer(1, 10))
            
            # Biometric Signals
            biometric_signals = analysis_result.get('biometric_signals', {})
            if biometric_signals:
                story.append(Paragraph("Biometric Signal Analysis", styles['Heading2']))
                
                biometric_data = [['Signal Type', 'Detection Level']]
                for signal, level in biometric_signals.items():
                    signal_name = signal.replace('_', ' ').title()
                    if isinstance(level, (int, float)):
                        level_str = f"{level:.2f}"
                    else:
                        level_str = str(level)
                    biometric_data.append([signal_name, level_str])
                
                biometric_table = Table(biometric_data, colWidths=[3*inch, 2*inch])
                biometric_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(biometric_table)
                story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("Recommendations", styles['Heading2']))
            recommendations = self._generate_recommendations(analysis_result)
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Footer
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
            story.append(Spacer(1, 30))
            story.append(Paragraph("Generated by Digital Deepfake Detection System", footer_style))
            story.append(Paragraph(f"Report ID: {task_id}", footer_style))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Generated PDF report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            raise
    
    async def generate_csv_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a CSV report with analysis data"""
        try:
            task_id = analysis_result.get('task_id', 'unknown')
            report_filename = f"deepfake_analysis_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow(['Digital Deepfake Detection Analysis Report'])
                writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['Analysis ID:', task_id])
                writer.writerow([])
                
                # Summary
                writer.writerow(['SUMMARY'])
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Analysis Type', analysis_result.get('analysis_type', 'Unknown')])
                writer.writerow(['Authenticity Score', f"{analysis_result.get('authenticity_score', 0)}%"])
                writer.writerow(['Confidence Level', analysis_result.get('confidence_level', 'Unknown')])
                writer.writerow(['Is Deepfake', analysis_result.get('detection_results', {}).get('is_deepfake', False)])
                writer.writerow(['Risk Level', analysis_result.get('detection_results', {}).get('risk_level', 'Unknown')])
                writer.writerow([])
                
                # File Info
                file_info = analysis_result.get('file_info', {})
                if file_info:
                    writer.writerow(['FILE INFORMATION'])
                    writer.writerow(['Property', 'Value'])
                    for key, value in file_info.items():
                        writer.writerow([key.replace('_', ' ').title(), value])
                    writer.writerow([])
                
                # Detailed Analysis
                detailed_analysis = analysis_result.get('detailed_analysis', {})
                if detailed_analysis:
                    writer.writerow(['DETAILED ANALYSIS'])
                    for category, data in detailed_analysis.items():
                        writer.writerow([category.replace('_', ' ').title().upper()])
                        if isinstance(data, dict):
                            writer.writerow(['Parameter', 'Value'])
                            for key, value in data.items():
                                if isinstance(value, list):
                                    value = f"{len(value)} items"
                                elif isinstance(value, (int, float)):
                                    value = f"{value:.4f}" if isinstance(value, float) else value
                                writer.writerow([key.replace('_', ' ').title(), value])
                        writer.writerow([])
                
                # Biometric Signals
                biometric_signals = analysis_result.get('biometric_signals', {})
                if biometric_signals:
                    writer.writerow(['BIOMETRIC SIGNALS'])
                    writer.writerow(['Signal Type', 'Detection Level'])
                    for signal, level in biometric_signals.items():
                        writer.writerow([signal.replace('_', ' ').title(), level])
                    writer.writerow([])
                
                # Recommendations
                recommendations = self._generate_recommendations(analysis_result)
                writer.writerow(['RECOMMENDATIONS'])
                for i, rec in enumerate(recommendations, 1):
                    writer.writerow([f"{i}.", rec])
            
            logger.info(f"Generated CSV report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"CSV generation error: {str(e)}")
            raise
    
    async def generate_batch_report(self, analysis_results: List[Dict[str, Any]]) -> str:
        """Generate a batch analysis report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"batch_deepfake_analysis_{timestamp}.pdf"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            doc = SimpleDocTemplate(report_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.darkblue,
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph("Batch Deepfake Detection Report", title_style))
            story.append(Spacer(1, 20))
            
            # Summary statistics
            total_files = len(analysis_results)
            deepfakes_detected = sum(1 for r in analysis_results if r.get('detection_results', {}).get('is_deepfake', False))
            avg_authenticity = np.mean([r.get('authenticity_score', 0) for r in analysis_results])
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Files Analyzed', total_files],
                ['Deepfakes Detected', deepfakes_detected],
                ['Detection Rate', f"{(deepfakes_detected/max(total_files, 1))*100:.1f}%"],
                ['Average Authenticity Score', f"{avg_authenticity:.1f}%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Individual results
            story.append(Paragraph("Individual Analysis Results", styles['Heading2']))
            
            results_data = [['File ID', 'Type', 'Authenticity Score', 'Deepfake?', 'Risk Level']]
            for result in analysis_results:
                results_data.append([
                    result.get('task_id', 'Unknown')[:10] + '...',
                    result.get('analysis_type', 'Unknown').title(),
                    f"{result.get('authenticity_score', 0):.1f}%",
                    'Yes' if result.get('detection_results', {}).get('is_deepfake', False) else 'No',
                    result.get('detection_results', {}).get('risk_level', 'Unknown')
                ])
            
            results_table = Table(results_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1*inch, 1.5*inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(results_table)
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Generated batch report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Batch report generation error: {str(e)}")
            raise
    
    def _get_score_assessment(self, score: float) -> str:
        """Get assessment text based on authenticity score"""
        if score >= 90:
            return "Highly Authentic"
        elif score >= 75:
            return "Likely Authentic"
        elif score >= 50:
            return "Uncertain"
        elif score >= 25:
            return "Likely Manipulated"
        else:
            return "Highly Manipulated"
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        authenticity_score = analysis_result.get('authenticity_score', 0)
        is_deepfake = analysis_result.get('detection_results', {}).get('is_deepfake', False)
        analysis_type = analysis_result.get('analysis_type', '')
        
        if is_deepfake or authenticity_score < 50:
            recommendations.append("High probability of manipulation detected. Exercise extreme caution.")
            recommendations.append("Verify content through multiple independent sources.")
            recommendations.append("Consider forensic analysis by digital media experts.")
        
        elif authenticity_score < 75:
            recommendations.append("Some irregularities detected. Further verification recommended.")
            recommendations.append("Cross-reference with original source if possible.")
        
        else:
            recommendations.append("Content appears authentic with high confidence.")
            recommendations.append("Standard verification practices still recommended.")
        
        # Type-specific recommendations
        if analysis_type == 'video':
            recommendations.append("For video content: Check for lip-sync accuracy and temporal consistency.")
        elif analysis_type == 'audio':
            recommendations.append("For audio content: Verify voice characteristics against known samples.")
        elif analysis_type == 'image':
            recommendations.append("For image content: Examine lighting, shadows, and facial geometry.")
        
        recommendations.append("Keep this analysis report for documentation and audit purposes.")
        
        return recommendations
    
    async def create_visualization_chart(self, data: Dict[str, Any], chart_type: str = 'score') -> str:
        """Create visualization charts for reports"""
        try:
            chart_filename = f"chart_{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = os.path.join(self.reports_dir, chart_filename)
            
            plt.figure(figsize=(10, 6))
            
            if chart_type == 'score':
                # Authenticity score visualization
                score = data.get('authenticity_score', 0)
                scores = [score, 100 - score]
                labels = ['Authentic', 'Suspicious']
                colors_list = ['green' if score > 50 else 'orange', 'red' if score <= 50 else 'lightgray']
                
                plt.pie(scores, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90)
                plt.title(f'Authenticity Analysis\nScore: {score}%')
            
            elif chart_type == 'timeline' and 'visualization_data' in data:
                # Timeline chart for video/audio
                viz_data = data['visualization_data']
                if 'timeline_data' in viz_data:
                    timeline = viz_data['timeline_data']
                    frames = [item.get('frame', 0) for item in timeline]
                    authenticity = [item.get('authenticity', 50) for item in timeline]
                    
                    plt.plot(frames, authenticity, 'b-', linewidth=2)
                    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Threshold')
                    plt.xlabel('Frame Number')
                    plt.ylabel('Authenticity Score (%)')
                    plt.title('Authenticity Over Time')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated chart: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Chart generation error: {str(e)}")
            return None
    
    def cleanup_old_reports(self, days: int = 7):
        """Clean up old report files"""
        try:
            current_time = datetime.now().timestamp()
            
            for filename in os.listdir(self.reports_dir):
                file_path = os.path.join(self.reports_dir, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if (current_time - file_time) > (days * 24 * 3600):
                        os.remove(file_path)
                        logger.info(f"Deleted old report: {filename}")
                        
        except Exception as e:
            logger.error(f"Report cleanup error: {str(e)}")
