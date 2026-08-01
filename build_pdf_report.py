import os
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import cm, inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Register fonts supporting Vietnamese & standard math glyphs
font_path_regular = "C:/Windows/Fonts/times.ttf"
font_path_bold = "C:/Windows/Fonts/timesbd.ttf"
font_path_italic = "C:/Windows/Fonts/timesi.ttf"
font_path_bolditalic = "C:/Windows/Fonts/timesbi.ttf"

pdfmetrics.registerFont(TTFont('Times-Roman', font_path_regular))
pdfmetrics.registerFont(TTFont('Times-Bold', font_path_bold))
pdfmetrics.registerFont(TTFont('Times-Italic', font_path_italic))
pdfmetrics.registerFont(TTFont('Times-BoldItalic', font_path_bolditalic))
pdfmetrics.registerFontFamily('Times-Roman', normal='Times-Roman', bold='Times-Bold', italic='Times-Italic', boldItalic='Times-BoldItalic')


class NumberedCanvasImpl(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvasImpl, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvasImpl, self).showPage()
        super(NumberedCanvasImpl, self).save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Skip header and footer on cover page
            return

        self.saveState()
        self.setFont("Times-Italic", 9)
        self.setFillColor(colors.HexColor("#333333"))

        # Header
        self.drawString(54, 842 - 36, "Báo cáo Môn học: Xử lý ngôn ngữ tự nhiên (INT3406 80) - SV: Nguyễn Nhật Minh (MSV: 24022406)")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 842 - 42, 595 - 54, 842 - 42)

        # Footer
        text = f"Trang {self._pageNumber} / {page_count}"
        self.drawRightString(595 - 54, 36, text)
        self.drawString(54, 36, "Trường Đại học Công nghệ - ĐHQGHN")
        self.line(54, 48, 595 - 54, 48)

        self.restoreState()


def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_cover_school = ParagraphStyle(
        'CoverSchool',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=13,
        leading=17,
        alignment=1, # Center
        textColor=colors.HexColor("#1A2B4C")
    )

    style_cover_subschool = ParagraphStyle(
        'CoverSubSchool',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.HexColor("#2C3E50")
    )

    style_cover_type = ParagraphStyle(
        'CoverType',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=16,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#B03A2E")
    )

    style_cover_title = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=20,
        leading=26,
        alignment=1,
        textColor=colors.HexColor("#11294D")
    )

    style_cover_subtitle = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=13,
        leading=18,
        alignment=1,
        textColor=colors.HexColor("#4A5568")
    )

    style_cover_info = ParagraphStyle(
        'CoverInfo',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=12,
        leading=20,
        textColor=colors.HexColor("#2D3748")
    )

    style_h1 = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=15,
        leading=20,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.HexColor("#1A365D"),
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12.5,
        leading=17,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#2B6CB0"),
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=16,
        spaceAfter=6,
        alignment=4, # Justify
        textColor=colors.HexColor("#2D3748")
    )

    style_bullet = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=16,
        spaceAfter=4,
        leftIndent=15,
        alignment=4,
        textColor=colors.HexColor("#2D3748")
    )

    style_table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2D3748")
    )

    style_table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1A365D")
    )

    style_table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=colors.white
    )

    style_code = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1A202C")
    )

    style_formula = ParagraphStyle(
        'FormulaStyle',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=11,
        leading=18,
        alignment=1, # Center
        textColor=colors.HexColor("#0D233A")
    )

    elements = []

    # Helper function to generate clean math formula blocks
    def create_formula_box(formulas, title=None):
        content = []
        if title:
            content.append(Paragraph(f"<b>{title}</b>", style_table_cell_bold))
            content.append(Spacer(1, 4))
        for f in formulas:
            content.append(Paragraph(f, style_formula))
            content.append(Spacer(1, 2))
        t = Table([[content]], colWidths=[487])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0F4F8")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BEE3F8")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        return t

    # ================= COVER PAGE =================
    elements.append(Paragraph("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ - ĐẠI HỌC QUỐC GIA HÀ NỘI", style_cover_school))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("KHOA CÔNG NGHỆ THÔNG TIN", style_cover_subschool))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A365D"), spaceBefore=0, spaceAfter=25))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("BÁO CÁO BÀI TẬP LỚN MÔN HỌC", style_cover_type))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("XỬ LÝ NGÔN NGỮ TỰ NHIÊN (INT3406 80)", style_cover_subtitle))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("ĐỀ TÀI: THỰC NGHIỆM VÀ SO SÁNH CÁC KIẾN TRÚC DỊCH MÁY TỰ ĐỘNG ANH - VIỆT (NMT SEQ2SEQ LUONG ATTENTION VÀ TRANSFORMER)", style_cover_title))
    elements.append(Spacer(1, 40))

    # Student Info Table Box (REMOVED folder name row as requested)
    info_data = [
        [Paragraph("<b>Môn học:</b>", style_cover_info), Paragraph("Xử lý ngôn ngữ tự nhiên", style_cover_info)],
        [Paragraph("<b>Mã lớp học phần:</b>", style_cover_info), Paragraph("INT3406 80", style_cover_info)],
        [Paragraph("<b>Họ và tên sinh viên:</b>", style_cover_info), Paragraph("Nguyễn Nhật Minh", style_cover_info)],
        [Paragraph("<b>Mã sinh viên:</b>", style_cover_info), Paragraph("24022406", style_cover_info)],
    ]
    info_table = Table(info_data, colWidths=[140, 280])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(info_table)

    elements.append(Spacer(1, 70))
    elements.append(Paragraph("Hà Nội - 2026", style_cover_subschool))
    elements.append(PageBreak())

    # ================= SECTION 1: TỔNG QUAN VÀ MỤC TIÊU =================
    elements.append(Paragraph("1. TỔNG QUAN VÀ MỤC TIÊU ĐỀ TÀI", style_h1))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=8))
    
    elements.append(Paragraph("<b>1.1. Đặt vấn đề</b>", style_h2))
    elements.append(Paragraph(
        "Dịch máy tự động (Neural Machine Translation - NMT) là một trong những bài toán trọng tâm và có tính ứng dụng cao nhất trong lĩnh vực Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP). Mục tiêu của bài toán là xây dựng mô hình mạng nơ-ron có khả năng chuyển đổi tự động một câu văn bản từ ngôn ngữ nguồn (English) sang ngôn ngữ đích (Vietnamese) mà vẫn giữ nguyên ngữ nghĩa, ngữ pháp và độ lưu khoát.",
        style_body
    ))
    elements.append(Paragraph(
        "Cùng với sự phát triển của học sâu (Deep Learning), các kiến trúc dịch máy đã có sự tiến hóa mạnh mẽ: từ các mô hình chuỗi sang chuỗi (Sequence-to-Sequence) dựa trên mạng nơ-ron hồi quy (RNN/LSTM) kết hợp cơ chế chú ý (Attention Mechanism), cho đến kiến trúc Transformer hoàn toàn dựa trên cơ chế Tự chú ý (Self-Attention) loại bỏ sự phụ thuộc tính toán tuần tự.",
        style_body
    ))

    elements.append(Paragraph("<b>1.2. Mục tiêu nghiên cứu và Yêu cầu thực nghiệm</b>", style_h2))
    elements.append(Paragraph("Trong bài tập lớn này, em tiến hành tái hiện (reimplement) và thực nghiệm so sánh chuyên sâu 2 kiến trúc mô hình NMT kinh điển trên bài toán dịch tự động Anh - Việt:", style_body))
    
    elements.append(Paragraph("• <b>Case 1: Mô hình Seq2Seq + Luong Attention</b>: Reimplement bằng thư viện PyTorch dựa trên kiến trúc và bộ siêu tham số (hparams <code>iwslt15.json</code>) của kho mã nguồn chuẩn <code>tensorflow/nmt</code> (Google Research). Mô hình sử dụng Encoder LSTM hai chiều (Bidirectional LSTM) và Decoder LSTM một chiều (Unidirectional LSTM) tích hợp cơ chế Scaled Luong Attention và Input Feeding.", style_bullet))
    elements.append(Paragraph("• <b>Case 2: Mô hình Transformer Encoder-Decoder</b>: Reimplement bằng thư viện PyTorch từ notebook hướng dẫn <code>demo_transformer.ipynb</code> (Google Colab). Mô hình dựa trên kiến trúc Transformer chuẩn (Attention Is All You Need) với 4 lớp Encoder-Decoder, Multi-Head Attention và Positional Encoding dạng hình sin.", style_bullet))

    elements.append(Paragraph("<b>1.3. Nguyên tắc so sánh công bằng (Fair Comparison)</b>", style_h2))
    elements.append(Paragraph(
        "Để đảm bảo tính khách quan và khoa học trong quá trình đánh giá, cả hai mô hình đều được huấn luyện và kiểm thử dưới cùng một điều kiện thực nghiệm khắt khe: (1) Cùng bộ dữ liệu song ngữ chuẩn IWSLT 2015 English-Vietnamese; (2) Cùng quy trình tiền xử lý và tách từ (tokenizer); (3) Cùng tập kiểm thử độc lập <code>tst2013</code>; (4) Cùng thuật toán giải mã (Greedy Search và Beam Search với beam_size = 5); (5) Đánh giá điểm BLEU tự động bằng cùng một module tính toán chuẩn.",
        style_body
    ))

    # ================= SECTION 2: BỘ DỮ LIỆU IWSLT 2015 =================
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("2. BỘ DỮ LIỆU THỰC NGHIỆM (IWSLT 2015 CORPUS)", style_h1))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=8))

    elements.append(Paragraph(
        "Bộ dữ liệu thực nghiệm được sử dụng là <b>IWSLT 2015 English-Vietnamese Parallel Corpus</b>, được trích xuất từ các bài nói chuyện TED Talks. Đây là bộ dữ liệu chuẩn mực được sử dụng phổ biến trong các nghiên cứu dịch máy Anh - Việt.",
        style_body
    ))

    elements.append(Paragraph("<b>2.1. Phân chia tập dữ liệu (Dataset Splits)</b>", style_h2))
    
    data_split_info = [
        [Paragraph("<b>Tập dữ liệu</b>", style_table_header), Paragraph("<b>Tên file</b>", style_table_header), Paragraph("<b>Số lượng cặp câu</b>", style_table_header), Paragraph("<b>Mục đích sử dụng</b>", style_table_header)],
        [Paragraph("Tập Huấn luyện (Train)", style_table_cell_bold), Paragraph("<code>train.en</code> / <code>train.vi</code>", style_table_cell), Paragraph("133,317 cặp câu", style_table_cell), Paragraph("Huấn luyện trọng số mô hình (Lọc <i>max_len</i> ≤ 80)", style_table_cell)],
        [Paragraph("Tập Phát triển (Validation)", style_table_cell_bold), Paragraph("<code>tst2012.en</code> / <code>tst2012.vi</code>", style_table_cell), Paragraph("1,553 cặp câu", style_table_cell), Paragraph("Theo dõi val loss và chọn checkpoint tốt nhất", style_table_cell)],
        [Paragraph("Tập Kiểm thử (Test)", style_table_cell_bold), Paragraph("<code>tst2013.en</code> / <code>tst2013.vi</code>", style_table_cell), Paragraph("1,268 cặp câu", style_table_cell), Paragraph("Đánh giá điểm BLEU độc lập và so sánh công bằng", style_table_cell)],
    ]
    table_split = Table(data_split_info, colWidths=[110, 110, 100, 167])
    table_split.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    elements.append(table_split)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>2.2. Tiền xử lý và Xây dựng Từ vựng (Vocabulary Building)</b>", style_h2))
    elements.append(Paragraph(
        "Quy trình xây dựng từ vựng được thực hiện nhất quán trên toàn bộ tập train với các tham số khống chế: tần suất xuất hiện tối thiểu (<code>min_freq = 2</code>) và kích thước từ vựng tối đa (<code>max_size = 30,000</code>). Kết quả xây dựng từ vựng thu được như sau:",
        style_body
    ))
    elements.append(Paragraph("• <b>Từ vựng tiếng Anh (EN Vocabulary)</b>: 28,145 từ duy nhất.", style_bullet))
    elements.append(Paragraph("• <b>Từ vựng tiếng Việt (VI Vocabulary)</b>: 12,234 từ duy nhất.", style_bullet))
    elements.append(Paragraph("• <b>Tập token đặc biệt (Special Tokens)</b>: <code>&lt;pad&gt;</code> (Chèn độ dài, ID=0), <code>&lt;unk&gt;</code> (Từ ngoài từ vựng, ID=1), <code>&lt;sos&gt;</code> (Bắt đầu chuỗi, ID=2), <code>&lt;eos&gt;</code> (Kết thúc chuỗi, ID=3).", style_bullet))

    # ================= SECTION 3: CASE 1 SEQ2SEQ =================
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("3. CHI TIẾT THỰC NGHIỆM CASE 1: NMT (SEQ2SEQ + LUONG ATTENTION)", style_h1))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=8))

    elements.append(Paragraph(
        "Case 1 tái hiện mô hình NMT kinh điển của Google (Luong et al., 2015) theo đúng thiết lập siêu tham số <code>iwslt15.json</code> từ kho mã nguồn <code>tensorflow/nmt</code> nhưng được cài đặt bằng PyTorch thuần.",
        style_body
    ))

    elements.append(Paragraph("<b>3.1. Kiến trúc Mô hình (Architecture Design)</b>", style_h2))
    elements.append(Paragraph("Mô hình bao gồm 3 thành phần chính:", style_body))
    elements.append(Paragraph("1. <b>Encoder (Bi-LSTM 2 lớp)</b>: Nhận chuỗi từ nguồn, đi qua lớp Embedding (<i>d</i><sub>embed</sub> = 512). Lớp LSTM thứ nhất chạy hai chiều (Bidirectional) với kích thước ẩn 256 cho mỗi chiều (tổng chiều ra = 512). Lớp LSTM thứ hai nhận đầu ra nối (concat) để tạo ra chuỗi trạng thái ẩn encoder <i>H</i> = [<i>h</i><sub>1</sub>, <i>h</i><sub>2</sub>, ..., <i>h</i><sub><i>T</i><sub><i>x</i></sub></sub>].", style_bullet))
    elements.append(Paragraph("2. <b>Decoder (Uni-LSTM 2 lớp)</b>: Nhận từ đích tại bước trước kết hợp với vector ngữ cảnh (Input Feeding). Decoder gồm 2 lớp LSTM một chiều với kích thước ẩn 512.", style_bullet))
    elements.append(Paragraph("3. <b>Scaled Luong Attention & Input Feeding</b>: Trọng số chú ý <i>a</i><sub><i>t</i></sub> và vector ngữ cảnh <i>c</i><sub><i>t</i></sub> được tính toán chi tiết theo các công thức toán học dưới đây.", style_bullet))

    elements.append(Spacer(1, 4))
    
    # Formatted Math Equations Box for Luong Attention
    math_seq2seq = [
        "<b>score</b>(<i>s</i><sub><i>t</i></sub>, <i>h</i><sub><i>i</i></sub>) = (<i>s</i><sub><i>t</i></sub><sup><i>T</i></sup> · <i>h</i><sub><i>i</i></sub>) / √<i>d</i>",
        "<i>a</i><sub><i>t,i</i></sub> = exp(<b>score</b>(<i>s</i><sub><i>t</i></sub>, <i>h</i><sub><i>i</i></sub>)) / ∑<sub><i>j</i></sub> exp(<b>score</b>(<i>s</i><sub><i>t</i></sub>, <i>h</i><sub><i>j</i></sub>))",
        "<i>c</i><sub><i>t</i></sub> = ∑<sub><i>i</i></sub> <i>a</i><sub><i>t,i</i></sub> <i>h</i><sub><i>i</i></sub>",
        "<i>s̃</i><sub><i>t</i></sub> = tanh(<i>W</i><sub><i>c</i></sub> [<i>c</i><sub><i>t</i></sub> ; <i>s</i><sub><i>t</i></sub>])"
    ]
    elements.append(create_formula_box(math_seq2seq, title="Công thức Toán học Cơ chế Scaled Luong Attention & Input Feeding"))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>3.2. Bảng Siêu tham số và Thông số Kỹ thuật Case 1</b>", style_h2))

    case1_params = [
        [Paragraph("<b>Thành phần / Siêu tham số</b>", style_table_header), Paragraph("<b>Giá trị cấu hình thực tế</b>", style_table_header)],
        [Paragraph("Kiến trúc Encoder", style_table_cell_bold), Paragraph("2-layer Bidirectional LSTM (<i>hidden_size</i> = 256/dir → 512)", style_table_cell)],
        [Paragraph("Kiến trúc Decoder", style_table_cell_bold), Paragraph("2-layer Unidirectional LSTM (<i>hidden_size</i> = 512)", style_table_cell)],
        [Paragraph("Kích thước Nhúng (Embedding Size)", style_table_cell_bold), Paragraph("512 (chia sẻ giữa Encoder và Decoder)", style_table_cell)],
        [Paragraph("Cơ chế Attention", style_table_cell_bold), Paragraph("Scaled Luong Attention (General/Dot) + Input Feeding", style_table_cell)],
        [Paragraph("Dropout Rate", style_table_cell_bold), Paragraph("0.2", style_table_cell)],
        [Paragraph("<b>Tổng số tham số (Total Parameters)</b>", style_table_cell_bold), Paragraph("<b>36,104,234 (36.1M parameters)</b>", style_table_cell_bold)],
        [Paragraph("Số Epochs huấn luyện", style_table_cell_bold), Paragraph("20 Epochs", style_table_cell)],
        [Paragraph("Optimizer & Learning Rate", style_table_cell_bold), Paragraph("Adam (<i>lr</i> = 0.001, decay theo epoch)", style_table_cell)],
        [Paragraph("Train Loss cuối cùng", style_table_cell_bold), Paragraph("<b>2.6823</b>", style_table_cell)],
        [Paragraph("Validation Loss tốt nhất (tst2012)", style_table_cell_bold), Paragraph("<b>3.3851</b>", style_table_cell)],
        [Paragraph("Tổng thời gian huấn luyện", style_table_cell_bold), Paragraph("<b>6,410.72 giây (~ 106.8 phút)</b>", style_table_cell)],
    ]
    table_c1 = Table(case1_params, colWidths=[200, 287])
    table_c1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    elements.append(table_c1)

    # ================= SECTION 4: CASE 2 TRANSFORMER =================
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("4. CHI TIẾT THỰC NGHIỆM CASE 2: TRANSFORMER ENCODER-DECODER", style_h1))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=8))

    elements.append(Paragraph(
        "Case 2 tái hiện kiến trúc Transformer chuẩn (Vaswani et al., 2017) dựa trên notebook <code>demo_transformer.ipynb</code> (PyTorch). Kiến trúc này loại bỏ hoàn toàn các lớp hồi quy RNN, thay vào đó dựa hoàn toàn vào các khối Tự chú ý đa đầu (Multi-Head Self-Attention).",
        style_body
    ))

    elements.append(Paragraph("<b>4.1. Kiến trúc Mô hình và Công thức Toán học (Architecture & Formulas)</b>", style_h2))
    elements.append(Paragraph("Mô hình bao gồm các khối chính được mô tả toán học như sau:", style_body))
    elements.append(Paragraph("1. <b>Positional Encoding (Sinusoidal)</b>: Mã hóa vị trí dựa trên hàm lượng giác sin và cos để giữ thông tin thứ tự từ:", style_bullet))
    
    math_pe = [
        "<i>PE</i><sub>(<i>pos</i>, 2<i>i</i>)</sub> = sin( <i>pos</i> / 10000<sup> 2<i>i</i> / <i>d</i><sub>model</sub> </sup> )",
        "<i>PE</i><sub>(<i>pos</i>, 2<i>i</i>+1)</sub> = cos( <i>pos</i> / 10000<sup> 2<i>i</i> / <i>d</i><sub>model</sub> </sup> )"
    ]
    elements.append(create_formula_box(math_pe, title="Công thức Mã hóa Vị trí (Sinusoidal Positional Encoding)"))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("2. <b>Scaled Dot-Product Attention & Multi-Head Attention</b>: Tính toán ma trận Tự chú ý và chú ý đa đầu:", style_bullet))
    
    math_attn = [
        "<b>Attention</b>(<i>Q</i>, <i>K</i>, <i>V</i>) = <b>softmax</b>( (<i>Q K</i><sup><i>T</i></sup>) / √<i>d</i><sub><i>k</i></sub> ) <i>V</i>",
        "<b>MultiHead</b>(<i>Q</i>, <i>K</i>, <i>V</i>) = <b>Concat</b>(<i>head</i><sub>1</sub>, ..., <i>head</i><sub><i>h</i></sub>) <i>W</i><sup><i>O</i></sup>",
        "trong đó: <i>head</i><sub><i>i</i></sub> = <b>Attention</b>(<i>Q W</i><sub><i>i</i></sub><sup><i>Q</i></sup>, <i>K W</i><sub><i>i</i></sub><sup><i>K</i></sup>, <i>V W</i><sub><i>i</i></sub><sup><i>V</i></sup>)"
    ]
    elements.append(create_formula_box(math_attn, title="Công thức Cơ chế Attention và Multi-Head Attention"))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("3. <b>Position-wise Feed-Forward Network (FFN)</b>: Biến đổi tuyến tính hai lớp qua hàm kích hoạt ReLU:", style_bullet))
    math_ffn = [
        "<b>FFN</b>(<i>x</i>) = <b>max</b>(0, <i>x W</i><sub>1</sub> + <i>b</i><sub>1</sub>) <i>W</i><sub>2</sub> + <i>b</i><sub>2</sub>"
    ]
    elements.append(create_formula_box(math_ffn, title="Công thức Khối Feed-Forward Network"))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>4.2. Bảng Siêu tham số và Thông số Kỹ thuật Case 2</b>", style_h2))

    case2_params = [
        [Paragraph("<b>Thành phần / Siêu tham số</b>", style_table_header), Paragraph("<b>Giá trị cấu hình thực tế</b>", style_table_header)],
        [Paragraph("Số lớp Encoder / Decoder (<i>N</i>)", style_table_cell_bold), Paragraph("4 Layers Encoder, 4 Layers Decoder", style_table_cell)],
        [Paragraph("Chiều ẩn mô hình (<i>d</i><sub>model</sub>)", style_table_cell_bold), Paragraph("512", style_table_cell)],
        [Paragraph("Chiều ẩn Feed-Forward (<i>d</i><sub>ff</sub>)", style_table_cell_bold), Paragraph("2048", style_table_cell)],
        [Paragraph("Số lượng Attention Heads (<i>h</i>)", style_table_cell_bold), Paragraph("8 heads (<i>d</i><sub><i>k</i></sub> = <i>d</i><sub><i>v</i></sub> = 512 / 8 = 64)", style_table_cell)],
        [Paragraph("Positional Encoding", style_table_cell_bold), Paragraph("Sinusoidal Positional Encoding (Fixed)", style_table_cell)],
        [Paragraph("Dropout Rate", style_table_cell_bold), Paragraph("0.1", style_table_cell)],
        [Paragraph("<b>Tổng số tham số (Total Parameters)</b>", style_table_cell_bold), Paragraph("<b>50,128,490 (50.1M parameters)</b>", style_table_cell_bold)],
        [Paragraph("Số Epochs huấn luyện", style_table_cell_bold), Paragraph("25 Epochs", style_table_cell)],
        [Paragraph("Optimizer & Schedule", style_table_cell_bold), Paragraph("Adam (β<sub>1</sub>=0.9, β<sub>2</sub>=0.98) + Warmup Learning Rate", style_table_cell)],
        [Paragraph("Train Loss cuối cùng", style_table_cell_bold), Paragraph("<b>0.8640</b>", style_table_cell)],
        [Paragraph("Validation Loss cuối cùng (tst2012)", style_table_cell_bold), Paragraph("<b>1.1684</b>", style_table_cell)],
        [Paragraph("Tổng thời gian huấn luyện", style_table_cell_bold), Paragraph("<b>1,778.74 giây (~ 29.6 phút)</b>", style_table_cell)],
    ]
    table_c2 = Table(case2_params, colWidths=[200, 287])
    table_c2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    elements.append(table_c2)

    # ================= SECTION 5: FAIR COMPARISON =================
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("5. KẾT QUẢ SO SÁNH VÀ ĐÁNH GIÁ CÔNG BẰNG (FAIR COMPARISON)", style_h1))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=8))

    elements.append(Paragraph(
        "Tất cả các mô hình được đánh giá lại trên tập kiểm thử độc lập <b>tst2013</b> (1,268 cặp câu). Điểm BLEU được đo lường tự động bằng thuật toán chuẩn hóa độ khớp n-gram với độ dài tham chiếu.",
        style_body
    ))

    elements.append(Paragraph("<b>5.1. Bảng So Sánh Tổng Hợp Kết Quả Thực Nghiệm</b>", style_h2))

    comparison_data = [
        [
            Paragraph("<b>Tiêu chí Đánh giá</b>", style_table_header),
            Paragraph("<b>Case 1: Seq2Seq + Luong Attn</b>", style_table_header),
            Paragraph("<b>Case 2: Transformer</b>", style_table_header),
            Paragraph("<b>Chênh lệch (Delta)</b>", style_table_header)
        ],
        [
            Paragraph("Kiến trúc cốt lõi", style_table_cell_bold),
            Paragraph("2-layer Bi-LSTM + Luong Attn", style_table_cell),
            Paragraph("4-layer Multi-Head Transformer", style_table_cell),
            Paragraph("Thay thế RNN bằng Self-Attn", style_table_cell)
        ],
        [
            Paragraph("Tổng số tham số", style_table_cell_bold),
            Paragraph("<b>36.1M</b> (36,104,234)", style_table_cell),
            Paragraph("<b>50.1M</b> (50,128,490)", style_table_cell),
            Paragraph("+ 14.0M params (+38.8%)", style_table_cell)
        ],
        [
            Paragraph("Số Epochs huấn luyện", style_table_cell_bold),
            Paragraph("20 epochs", style_table_cell),
            Paragraph("25 epochs", style_table_cell),
            Paragraph("+ 5 epochs", style_table_cell)
        ],
        [
            Paragraph("Train Loss cuối cùng", style_table_cell_bold),
            Paragraph("2.6823", style_table_cell),
            Paragraph("<b>0.8640</b>", style_table_cell_bold),
            Paragraph("Transformer hội tụ sâu hơn", style_table_cell)
        ],
        [
            Paragraph("Validation Loss (tst2012)", style_table_cell_bold),
            Paragraph("3.3851", style_table_cell),
            Paragraph("<b>1.1684</b>", style_table_cell_bold),
            Paragraph("Transformer ít overfit hơn", style_table_cell)
        ],
        [
            Paragraph("<b>BLEU Greedy (tst2013)</b>", style_table_cell_bold),
            Paragraph("27.49", style_table_cell),
            Paragraph("<b>28.81</b>", style_table_cell_bold),
            Paragraph("<b>+ 1.32 BLEU</b>", style_table_cell_bold)
        ],
        [
            Paragraph("<b>BLEU Beam Search (beam=5)</b>", style_table_cell_bold),
            Paragraph("28.71", style_table_cell),
            Paragraph("<b>30.00</b>", style_table_cell_bold),
            Paragraph("<b>+ 1.29 BLEU (Đạt mốc 30.0)</b>", style_table_cell_bold)
        ],
        [
            Paragraph("<b>Thời gian huấn luyện</b>", style_table_cell_bold),
            Paragraph("6,410.72s (~ 106.8 phút)", style_table_cell),
            Paragraph("<b>1,778.74s (~ 29.6 phút)</b>", style_table_cell_bold),
            Paragraph("<b>Nhanh hơn 3.6x (giảm 72.3%)</b>", style_table_cell_bold)
        ],
    ]

    table_comp = Table(comparison_data, colWidths=[130, 130, 130, 97])
    table_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    elements.append(table_comp)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>5.2. Phân tích Chi tiết Kết quả So sánh</b>", style_h2))
    elements.append(Paragraph(
        "1. <b>Chất lượng dịch thuật (BLEU Score)</b>: Transformer vượt trội hoàn toàn so với Seq2Seq + Luong Attention ở cả hai chế độ giải mã. Khi áp dụng Beam Search (beam_size = 5), Transformer đạt điểm BLEU vượt ngưỡng 30.00 điểm (so với 28.71 của Seq2Seq), tạo khoảng cách chênh lệch +1.29 điểm BLEU. Kết quả này chứng minh khả năng học các mối quan hệ ngữ nghĩa xa (long-range dependencies) xuất sắc của cơ chế Self-Attention.",
        style_body
    ))
    elements.append(Paragraph(
        "2. <b>Tốc độ và Hiệu năng tính toán</b>: Mặc dù Transformer có dung lượng tham số lớn hơn (50.1M so với 36.1M), thời gian huấn luyện của Transformer lại <b>nhanh hơn gấp 3.6 lần</b> (chỉ mất 29.6 phút so với 106.8 phút của Seq2Seq). Nguyên nhân là do mạng LSTM của Seq2Seq bắt buộc phải tính toán tuần tự từng bước thời gian t, trong khi Transformer có thể tính toán song song hóa ma trận chú ý trên toàn bộ chuỗi đầu vào.",
        style_body
    ))
    elements.append(Paragraph(
        "3. <b>Khả năng hội tụ</b>: Train loss của Transformer đạt mức rất thấp (0.8640 so với 2.6823 của Seq2Seq), đồng thời Val loss duy trì ở mức 1.1684 mà không bị lặp hiện tượng quá khớp (overfitting).",
        style_body
    ))

    # ================= SECTION 6: PHÂN TÍCH VÍ DỤ DỊCH THỰC TẾ =================
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("6. PHÂN TÍCH VÍ DỤ DỊCH THỰC TẾ (QUALITATIVE ANALYSIS)", style_h1))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=8))

    elements.append(Paragraph(
        "Dưới đây là 5 ví dụ trích xuất thực tế từ tập kiểm thử <b>tst2013</b> với chế độ giải mã Beam Search (beam_size = 5), so sánh trực tiếp kết quả dịch của Case 1 (Seq2Seq) và Case 2 (Transformer) với câu gốc (Input) và bản dịch tham chiếu (Reference).",
        style_body
    ))

    examples = [
        {
            "id": "Ví dụ 1",
            "en": 'When I was little , I thought my country was the best on the planet , and I grew up singing a song called " Nothing To Envy . "',
            "ref": 'Khi tôi còn nhỏ , Tôi nghĩ rằng BắcTriều Tiên là đất nước tốt nhất trên thế giới và tôi thường hát bài " Chúng ta chẳng có gì phải ghen tị . "',
            "seq": 'khi tôi còn nhỏ , tôi nghĩ đất nước mình là người giỏi nhất hành tinh , và tôi lớn lên hát một bài hát tên là " không có gì để ghen tị . "',
            "trans": 'khi tôi còn nhỏ , tôi nghĩ đất nước tôi là người giỏi nhất trên hành tinh , và tôi lớn lên hát một bài hát có tên là " không có gì để ghen tị " .'
        },
        {
            "id": "Ví dụ 2",
            "en": 'And I was very proud .',
            "ref": 'Tôi đã rất tự hào về đất nước tôi .',
            "seq": 'và tôi đã rất tự hào .',
            "trans": 'và tôi rất tự hào .'
        },
        {
            "id": "Ví dụ 3",
            "en": 'In school , we spent a lot of time studying the history of Kim Il-Sung , but we never learned much about the outside world , except that America , South Korea , Japan are the enemies .',
            "ref": 'Ở trường , chúng tôi dành rất nhiều thời gian để học về cuộc đời của chủ tịch Kim II- Sung , nhưng lại không học nhiều về thế giới bên ngoài , ngoại trừ việc Hoa Kỳ , Hàn Quốc và Nhật Bản là kẻ thù của chúng tôi .',
            "seq": 'ở trường , chúng tôi đã dành rất nhiều thời gian nghiên cứu lịch sử của kim <unk> , nhưng chúng tôi chưa bao giờ học được nhiều về thế giới bên ngoài , ngoại trừ nước mỹ , nam hàn , nhật bản là kẻ thù .',
            "trans": 'ở trường , chúng tôi đã dành rất nhiều thời gian nghiên cứu lịch sử của kim <unk> , nhưng chúng tôi chưa bao giờ học được nhiều về thế giới bên ngoài , ngoại trừ nước mỹ , hàn quốc , nhật bản là những kẻ thù .'
        },
        {
            "id": "Ví dụ 4",
            "en": 'Although I often wondered about the outside world , I thought I would spend my entire life in North Korea , until everything suddenly changed .',
            "ref": 'Mặc dù tôi đã từng tự hỏi không biết thế giới bên ngoài kia như thế nào , nhưng tôi vẫn nghĩ rằng mình sẽ sống cả cuộc đời ở BắcTriều Tiên , cho tới khi tất cả mọi thứ đột nhiên thay đổi .',
            "seq": 'mặc dù tôi thường tự hỏi về thế giới bên ngoài , tôi nghĩ tôi sẽ dành cả đời mình ở bắc triều tiên , cho đến mọi thứ đột ngột thay đổi .',
            "trans": 'mặc dù tôi thường tự hỏi về thế giới bên ngoài , tôi nghĩ mình sẽ dành toàn bộ cuộc sống của mình ở bắc triều tiên , cho đến khi mọi thứ đột nhiên thay đổi .'
        },
        {
            "id": "Ví dụ 5",
            "en": 'When I was seven years old , I saw my first public execution , but I thought my life in North Korea was normal .',
            "ref": 'Khi tôi lên 7 , tôi chứng kiến cảnh người ta xử bắn công khai lần đầu tiên trong đời , nhưng tôi vẫn nghĩ cuộc sống của mình ở đây là hoàn toàn bình thường .',
            "seq": 'khi tôi bảy tuổi , tôi thấy cuộc hành trình công cộng đầu tiên của mình , nhưng tôi nghĩ cuộc sống của mình ở bắc triều tiên là bình thường .',
            "trans": 'khi tôi 7 tuổi , tôi thấy cuộc hành hình công cộng đầu tiên của mình , nhưng tôi nghĩ cuộc sống của tôi ở bắc triều tiên là bình thường .'
        }
    ]

    for ex in examples:
        ex_table_data = [
            [Paragraph(f"<b>{ex['id']}</b>", style_table_header), Paragraph("", style_table_header)],
            [Paragraph("<b>Input (EN):</b>", style_table_cell_bold), Paragraph(ex['en'], style_code)],
            [Paragraph("<b>Reference (VI):</b>", style_table_cell_bold), Paragraph(ex['ref'], style_code)],
            [Paragraph("<b>Case 1 (Seq2Seq):</b>", style_table_cell_bold), Paragraph(ex['seq'], style_code)],
            [Paragraph("<b>Case 2 (Transformer):</b>", style_table_cell_bold), Paragraph(ex['trans'], style_code)],
        ]
        t_ex = Table(ex_table_data, colWidths=[120, 367])
        t_ex.setStyle(TableStyle([
            ('SPAN', (0,0), (1,0)),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ]))
        elements.append(KeepTogether([t_ex, Spacer(1, 8)]))

    elements.append(Paragraph("<b>Nhận xét chất lượng dịch qua các ví dụ thực tế:</b>", style_h2))
    elements.append(Paragraph("• <b>Độ chính xác về từ vựng và ngữ pháp</b>: Ở Ví dụ 4, bản dịch của Transformer <i>'cho đến khi mọi thứ đột nhiên thay đổi'</i> trôi chảy và đúng ngữ pháp hơn hẳn Seq2Seq <i>'cho đến mọi thứ đột ngột thay đổi'</i> (thiếu quan hệ từ 'khi'). Ở Ví dụ 5, Transformer dịch đúng thuật ngữ <i>'cuộc hành hình công cộng'</i> (public execution), trong khi Seq2Seq dịch nhầm thành <i>'cuộc hành trình công cộng'</i>.", style_bullet))
    elements.append(Paragraph("• <b>Xử lý từ hiếm (&lt;unk&gt;)</b>: Các từ tên riêng phức tạp như <i>Kim Il-Sung</i> chưa có trong từ vựng min_freq=2 đều bị gán token <code>&lt;unk&gt;</code> ở cả 2 mô hình (Ví dụ 3), gợi mở nhu cầu cần áp dụng kỹ thuật mã hóa từ con BPE (Byte Pair Encoding).", style_bullet))

    # ================= SECTION 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN =================
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", style_h1))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=8))

    elements.append(Paragraph("<b>7.1. Kết luận tổng quan</b>", style_h2))
    elements.append(Paragraph(
        "Thông qua quá trình cài đặt thực nghiệm và so sánh công bằng trên bộ dữ liệu IWSLT 2015 English-Vietnamese, bài báo cáo đã rút ra các kết luận quan trọng sau:",
        style_body
    ))
    elements.append(Paragraph("1. <b>Transformer áp đảo về chất lượng dịch</b>: Với điểm BLEU Beam Search đạt <b>30.00 điểm</b> so với 28.71 điểm của Seq2Seq + Luong Attention, Transformer khẳng định tính hiệu quả vượt trội trong việc nắm bắt ngữ cảnh và phụ thuộc xa.", style_bullet))
    elements.append(Paragraph("2. <b>Transformer áp đảo về tốc độ huấn luyện</b>: Nhờ cơ chế tính toán song song hóa loại bỏ ma trận tính toán tuần tự thời gian của LSTM, Transformer tiết kiệm <b>72.3% thời gian huấn luyện</b> (chỉ mất 29.6 phút so với 106.8 phút của Seq2Seq).", style_bullet))
    elements.append(Paragraph("3. <b>Giải mã Beam Search có hiệu quả rõ rệt</b>: Ở cả hai mô hình, chiến lược giải mã Beam Search (beam_size = 5) đều giúp tăng từ +1.19 đến +1.22 điểm BLEU so với giải mã tham lam (Greedy Search).", style_bullet))

    elements.append(Paragraph("<b>7.2. Hướng phát triển tiếp theo (Future Work)</b>", style_h2))
    elements.append(Paragraph("Để tiếp tục nâng cao chất lượng mô hình dịch máy Anh - Việt, các hướng nghiên cứu mở rộng bao gồm:", style_body))
    elements.append(Paragraph("• <b>Áp dụng Tách từ con (Subword Tokenization)</b>: Thay thế phương pháp tách từ theo khoảng trắng bằng BPE (Byte Pair Encoding) hoặc WordPiece để giải quyết triệt để vấn đề từ ngoài từ vựng (<code>&lt;unk&gt;</code>) đối với tên riêng và từ ghép tiếng Việt.", style_bullet))
    elements.append(Paragraph("• <b>Tận dụng Mô hình Tiền huấn luyện (Pre-trained LLMs / NMT)</b>: Tinh chỉnh (fine-tune) các mô hình đa ngôn ngữ lớn như mBART-50, NMT-M2M100 hoặc PhoBERT/VietNamese-LLM để đạt điểm BLEU vượt trội.", style_bullet))
    elements.append(Paragraph("• <b>Mở rộng Bộ dữ liệu huấn luyện</b>: Tích hợp thêm các bộ dữ liệu dịch Anh - Việt lớn như OPUS, PhoMT để gia tăng độ phong phú ngữ cảnh cho mô hình.", style_bullet))

    # Build PDF with NumberedCanvas
    doc.build(elements, canvasmaker=NumberedCanvasImpl)
    print(f"PDF successfully generated at: {filename}")

if __name__ == "__main__":
    out_pdf = "c:/Users/admin/Documents/GitHub/24022406/Bao_Cao_XNLTN_24022406.pdf"
    build_pdf(out_pdf)
