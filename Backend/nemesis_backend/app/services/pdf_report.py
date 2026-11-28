"""
PDF Report Generator for NEMESIS-X Incident Reports
Creates professional PDF reports with evidence screenshots from tracked video.
"""

import cv2
import os
import json
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
from typing import List, Dict, Any, Optional
import numpy as np


class PDFReportGenerator:
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            # Anchor to the backend package dir so the write location matches
            # the absolute path the download endpoint reads from, regardless of CWD.
            backend_dir = Path(__file__).resolve().parent.parent.parent
            output_dir = str(backend_dir / "output" / "reports")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames_from_video(self, video_path: str, frame_ids: List[int],
                                   timestamps: List[float], fps: float) -> Dict[int, str]:
        """
        Extract specific frames from the tracked video and save as images.
        Returns dict mapping frame_id to saved image path.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return {}

        frame_images = {}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in frame_ids:
                # Save frame as image
                image_path = self.output_dir / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(image_path), frame)
                frame_images[frame_count] = str(image_path)
                print(f"Extracted frame {frame_count} at {timestamps[frame_ids.index(frame_count)]:.2f}s")

            frame_count += 1

        cap.release()
        return frame_images

    def generate_report_pdf(self, alert, frames, video_path: str, video_fps: float,
                             video_width: int, video_height: int) -> str:
        """
        Generate a professional PDF report with screenshots.
        Returns the path to the generated PDF.
        """
        # Get evidence frames from alert
        evidence_frames = alert.evidence_frames
        evidence_timestamps = [f.timestamp for f in frames if f.frame_id in evidence_frames]

        # Extract frames from video
        frame_images = self.extract_frames_from_video(video_path, evidence_frames, evidence_timestamps, video_fps)

        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        # ============ COVER PAGE ============
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 28)
        pdf.ln(40)
        pdf.cell(0, 15, "NEMESIS-X", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 18)
        pdf.cell(0, 12, "INCIDENT REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)

        # Divider line
        pdf.set_draw_color(0, 102, 204)
        pdf.set_line_width(1)
        pdf.line(60, pdf.get_y(), 150, pdf.get_y())
        pdf.ln(15)

        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Event ID: {alert.event_id}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Video ID: {alert.video_id}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Anomaly Score: {alert.anomaly_score:.2f}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(20)

        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 6, "Anomaly scores indicate deviation from a learned baseline;\nthey are not probabilities of crime, intent, or danger.\nHuman review is required.", align="C")

        # ============ INCIDENT SUMMARY ============
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "1. INCIDENT SUMMARY", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(0, 102, 204)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        summary = (
            f"An anomalous pattern was detected in video '{alert.video_id}' "
            f"with an anomaly score of {alert.anomaly_score:.2f} "
            f"between {alert.start_time:.1f}s and {alert.end_time:.1f}s. "
            f"The detection was triggered by sudden kinematic deviation or spatial "
            f"divergence from the learned baseline involving {len(alert.involved_track_ids)} "
            f"tracked entities."
        )
        pdf.multi_cell(0, 6, summary)
        pdf.ln(3)

        # ============ INCIDENT DETAILS ============
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "2. INCIDENT DETAILS", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(0, 102, 204)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        details = [
            ("Video ID", alert.video_id),
            ("Event ID", alert.event_id),
            ("Anomaly Score", f"{alert.anomaly_score:.3f}"),
            ("Start Time", f"{alert.start_time:.2f} seconds"),
            ("End Time", f"{alert.end_time:.2f} seconds"),
            ("Duration", f"{alert.end_time - alert.start_time:.2f} seconds"),
            ("Involved Track IDs", ", ".join(map(str, alert.involved_track_ids))),
            ("Evidence Frames", ", ".join(map(str, alert.evidence_frames))),
        ]

        pdf.set_font("Helvetica", "", 11)
        for label, value in details:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(55, 8, f"{label}:", new_x="END")
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, str(value), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # ============ EVIDENCE SCREENSHOTS ============
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "3. EVIDENCE SCREENSHOTS", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(0, 102, 204)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6,
            "Key frames from the tracked video at the time of the anomaly. "
            "Frames show detected objects with bounding boxes and track IDs."
        )
        pdf.ln(5)

        # Add evidence frames in a grid
        y_start = pdf.get_y()
        max_images_per_row = 2
        img_width = 90
        img_height = 55
        x_margin = 10
        y_margin = 10

        for idx, frame_id in enumerate(evidence_frames):
            if frame_id in frame_images:
                col = idx % max_images_per_row
                row = idx // max_images_per_row

                x = x_margin + col * (img_width + 10)
                y = y_start + row * (img_height + 15)

                # Check if we need a new page
                if y + img_height > 270:
                    pdf.add_page()
                    y_start = 20
                    y = y_start + row * (img_height + 15)
                    x = x_margin + col * (img_width + 10)

                # Draw image
                try:
                    pdf.image(frame_images[frame_id], x=x, y=y, w=img_width, h=img_height)
                except Exception as e:
                    print(f"Warning: Could not add image for frame {frame_id}: {e}")
                    pdf.set_xy(x, y)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.cell(img_width, img_height, f"Frame {frame_id}\n(Image unavailable)", border=1, align="C")

                # Frame label
                ts = evidence_timestamps[idx] if idx < len(evidence_timestamps) else 0
                pdf.set_xy(x, y + img_height + 2)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(img_width, 5, f"Frame {frame_id}  |  t={ts:.2f}s", align="C")

        pdf.ln(30)

        # ============ OBJECT AND TRACK ANALYSIS ============
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "4. OBJECT AND TRACK ANALYSIS", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(0, 102, 204)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Gather track info from frames
        track_info = {}
        for frame in frames:
            for det in frame.detections:
                if det.track_id in alert.involved_track_ids:
                    if det.track_id not in track_info:
                        track_info[det.track_id] = {
                            "label": det.label,
                            "first_seen": frame.timestamp,
                            "last_seen": frame.timestamp,
                            "frames": []
                        }
                    track_info[det.track_id]["last_seen"] = frame.timestamp
                    track_info[det.track_id]["frames"].append(frame.frame_id)

        pdf.set_font("Helvetica", "", 11)
        for track_id, info in sorted(track_info.items()):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, f"Track ID: {track_id}  ({info['label']})", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"  First seen: {info['first_seen']:.2f}s  |  Last seen: {info['last_seen']:.2f}s  |  Duration: {info['last_seen'] - info['first_seen']:.2f}s  |  Frames: {len(info['frames'])}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        pdf.ln(5)

        # ============ TEMPORAL AND MOVEMENT ANALYSIS ============
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "5. TEMPORAL AND MOVEMENT ANALYSIS", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(0, 102, 204)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Calculate displacement and speed for each track
        for track_id, info in sorted(track_info.items()):
            centers = []
            for frame in frames:
                for det in frame.detections:
                    if det.track_id == track_id:
                        x1, y1, x2, y2 = det.bbox
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        centers.append((frame.timestamp, cx, cy))

            if len(centers) >= 2:
                first_t, first_x, first_y = centers[0]
                last_t, last_x, last_y = centers[-1]
                displacement = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
                duration = last_t - first_t
                speed = displacement / duration if duration > 0 else 0

                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, f"Track {track_id}:", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 5, f"  Displacement: {displacement:.4f} (normalized)  |  Speed: {speed:.4f}  |  Observations: {len(centers)}")
                pdf.ln(2)

        pdf.ln(5)

        # ============ AI ASSESSMENT ============
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "6. AI ASSESSMENT", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(0, 102, 204)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        assessment = (
            f"The anomaly score of {alert.anomaly_score:.2f} indicates a deviation from the "
            f"learned baseline of normal activity for this scene. The following observations "
            f"are based solely on the tracking evidence:\n\n"
            f"- {len(alert.involved_track_ids)} tracked entities were active during the anomaly window\n"
            f"- The event occurred between {alert.start_time:.1f}s and {alert.end_time:.1f}s\n"
            f"- Evidence frames: {', '.join(map(str, alert.evidence_frames))}\n\n"
            f"Important: This score represents statistical deviation from baseline patterns. "
            f"It does not indicate criminal intent, suspicious behavior, or any specific "
            f"activity classification. Human review is required to determine the nature "
            f"and significance of the observed pattern."
        )
        pdf.multi_cell(0, 6, assessment)
        pdf.ln(5)

        # ============ RECOMMENDED ACTIONS ============
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "7. RECOMMENDED ACTIONS", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(0, 102, 204)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        actions = [
            "Review the evidence screenshots and tracking data for the indicated time window",
            "Cross-reference with other camera feeds if available",
            "Verify if the anomaly corresponds to a known event or expected activity",
            "Consider adjusting the anomaly threshold or calibration period if false positives occur",
            "Archive this report for audit trail and pattern analysis"
        ]

        pdf.set_font("Helvetica", "", 11)
        for i, action in enumerate(actions, 1):
            pdf.cell(5, 6, f"{i}.", new_x="END")
            pdf.multi_cell(0, 6, action)
            pdf.ln(1)

        pdf.ln(10)

        # ============ FOOTER ============
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 5, "NEMESIS-X Temporal Intelligence System", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, "This is an automated analysis. Human review required.", align="C", new_x="LMARGIN", new_y="NEXT")

        # Save PDF
        pdf_filename = f"{alert.video_id}_{alert.event_id}_report.pdf"
        pdf_path = self.output_dir / pdf_filename
        pdf.output(str(pdf_path))
        print(f"PDF report saved to: {pdf_path}")

        return str(pdf_path)


def generate_pdf_report(alert, frames, video_path: str, video_fps: float,
                         video_width: int, video_height: int) -> str:
    """Convenience function to generate PDF report."""
    generator = PDFReportGenerator()
    return generator.generate_report_pdf(alert, frames, video_path, video_fps, video_width, video_height)