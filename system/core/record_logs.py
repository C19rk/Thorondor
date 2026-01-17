import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import threading
import queue
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

class LogRecorder:
    def __init__(self, max_queue=1000):
        self.recording = False
        self.output_dir = os.path.normpath("logs")
        self.filename = None
        self.log_queue = queue.Queue(maxsize=max_queue)
        self.worker = None
        self.lock = threading.Lock()
        self._stop_signal = object()  # sentinel to stop thread
        self.directory_set = False
        self.log_entries = []  # Store log entries for PDF generation

    def set_directory_popup(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_dir = filedialog.askdirectory(title="Select Log Save Directory")
        root.destroy()
        if selected_dir:
            self.output_dir = os.path.normpath(selected_dir)
            os.makedirs(self.output_dir, exist_ok=True)
            self.directory_set = True
            return self.output_dir
        return None

    def start(self):
        if self.recording:
            return
        self.recording = True
        self.log_entries = []  # Clear previous entries
        # Ensure directory exists before recording
        os.makedirs(self.output_dir, exist_ok=True)
        with self.log_queue.mutex:
            self.log_queue.queue.clear()
        
        # Create filename with user-friendly format: "Argus Report Log - Jan 17, 2026.pdf"
        now = datetime.now()
        month_name = now.strftime("%b")  # Short month name (Jan, Feb, etc.)
        day = now.strftime("%d")
        year = now.strftime("%Y")
        friendly_name = f"Argus Report Log - {month_name} {day}, {year}.pdf"
        self.filename = os.path.join(self.output_dir, friendly_name)
        
        self.worker = threading.Thread(target=self._record_worker, daemon=True)
        self.worker.start()
        print(f"[INFO] Log recording started: {self.filename}")

    def _record_worker(self):
        start_time = datetime.now()
        
        while self.recording or not self.log_queue.empty():
            try:
                log_entry = self.log_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if log_entry is self._stop_signal:
                continue

            # Store log entry for PDF generation
            self.log_entries.append(log_entry)
        
        # Generate PDF after recording stops
        self._generate_pdf(start_time)
        print(f"[INFO] Log recording stopped and saved: {self.filename}")

    def _generate_pdf(self, start_time):
        """Generate a user-friendly PDF report"""
        doc = SimpleDocTemplate(self.filename, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='darkblue',
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Custom heading style
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='darkblue',
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Custom body style
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            fontName='Courier',
            leftIndent=20
        )
        
        # Add title
        title = Paragraph("ARGUS DETECTION REPORT", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Add metadata
        stop_time = datetime.now()
        duration = stop_time - start_time
        
        metadata = f"""
        <b>Report Generated:</b> {stop_time.strftime("%B %d, %Y at %I:%M:%S %p")}<br/>
        <b>Recording Started:</b> {start_time.strftime("%B %d, %Y at %I:%M:%S %p")}<br/>
        <b>Recording Stopped:</b> {stop_time.strftime("%B %d, %Y at %I:%M:%S %p")}<br/>
        <b>Duration:</b> {str(duration).split('.')[0]}<br/>
        <b>Total Events:</b> {len(self.log_entries)}
        """
        
        elements.append(Paragraph(metadata, styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))
        
        # Add detection log section
        elements.append(Paragraph("DETECTION LOG", heading_style))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Add log entries
        if self.log_entries:
            for entry in self.log_entries:
                # Escape special characters for XML
                safe_entry = entry.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(safe_entry, body_style))
                elements.append(Spacer(1, 0.05 * inch))
        else:
            elements.append(Paragraph("No detection events recorded.", styles['Normal']))
        
        # Build PDF
        doc.build(elements)

    def write(self, log_message):
        """Add a log message to the queue"""
        if self.recording:
            try:
                self.log_queue.put_nowait(log_message)
            except queue.Full:
                pass  # drop logs to prevent blocking

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        print("[INFO] Stopping log recording...")
        # Wait for worker to finish
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=5)