const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Nhom 15";
pres.title = "Nhan dang bien so xe do phan giai thap - ResTranOCR";

// ---------- Palette ----------
const NAVY = "16213E";
const SLATE = "2D4263";
const AMBER = "F2C14E";
const LIGHT = "F4F6FA";
const CARD = "FFFFFF";
const TEXT_DARK = "1B2A4A";
const MUTED = "6B7A99";
const TEAL = "3E7CB1";

const W = 13.333, H = 7.5;

// ---------- Helpers ----------
function shadow() {
    return { type: "outer", color: "16213E", blur: 8, offset: 3, angle: 45, opacity: 0.12 };
}

function pageNumber(slide, n) {
    slide.addText(String(n), {
        x: W - 0.7, y: H - 0.55, w: 0.5, h: 0.35,
        fontSize: 11, color: MUTED, align: "right", fontFace: "Calibri",
    });
}

function lightSlide(kicker, title) {
    const slide = pres.addSlide();
    slide.background = { color: LIGHT };
    slide.addText(kicker.toUpperCase(), {
        x: 0.7, y: 0.4, w: 10, h: 0.35,
        fontSize: 13, color: TEAL, bold: true, fontFace: "Calibri", charSpacing: 2,
    });
    slide.addText(title, {
        x: 0.7, y: 0.7, w: 12, h: 0.8,
        fontSize: 30, color: TEXT_DARK, bold: true, fontFace: "Cambria",
    });
    return slide;
}

function iconPath(name, color) {
    return `icons/${name}_${color}.png`;
}

function iconCircle(slide, name, x, y, d, bg, iconColor) {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg } });
    const pad = d * 0.24;
    slide.addImage({ path: iconPath(name, iconColor), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
}

// =====================================================================
// SLIDE 1 - TITLE
// =====================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: NAVY };

    // license-plate motif shapes
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 9.6, y: 0.9, w: 3.0, h: 0.95, rectRadius: 0.12,
        fill: { color: AMBER }, line: { color: "FFFFFF", width: 2 },
    });
    slide.addText("AVL 5215", {
        x: 9.6, y: 0.9, w: 3.0, h: 0.95, align: "center", valign: "middle",
        fontSize: 24, bold: true, color: NAVY, fontFace: "Courier New", charSpacing: 3,
    });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 10.05, y: 2.05, w: 2.4, h: 0.7, rectRadius: 0.1,
        fill: { color: SLATE }, line: { color: AMBER, width: 1.5 },
    });
    slide.addText("SFG 1D23", {
        x: 10.05, y: 2.05, w: 2.4, h: 0.7, align: "center", valign: "middle",
        fontSize: 18, bold: true, color: "FFFFFF", fontFace: "Courier New", charSpacing: 3,
    });

    slide.addText("ICPR 2026 · LOW-RESOLUTION LICENSE PLATE RECOGNITION CHALLENGE", {
        x: 0.8, y: 1.0, w: 8.5, h: 0.4,
        fontSize: 13, color: AMBER, bold: true, fontFace: "Calibri", charSpacing: 2,
    });

    slide.addText("Xây dựng hệ thống nhận dạng biển số xe\nđộ phân giải thấp dựa trên ResTranOCR\nvà Multi-frame Attention Fusion", {
        x: 0.8, y: 1.6, w: 9.0, h: 2.6,
        fontSize: 40, color: "FFFFFF", bold: true, fontFace: "Cambria", lineSpacingMultiple: 1.05,
    });

    slide.addText("Dự án: LowResolution-LPR", {
        x: 0.8, y: 4.35, w: 8, h: 0.4,
        fontSize: 16, color: "CADCFC", italic: true, fontFace: "Calibri",
    });

    // divider area
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 0, y: 5.35, w: W, h: 0.01, fill: { color: "2D4263" },
    });

    slide.addText([
        { text: "Học viện Công nghệ Bưu chính Viễn thông — Khoa Công nghệ Thông tin 1", options: { breakLine: true, color: "CADCFC" } },
        { text: "Giảng viên hướng dẫn: TS. Nguyễn Xuân Đức", options: { breakLine: true, color: "CADCFC" } },
        { text: "Nhóm 15: Phạm Văn Kiên (B22DCKH063)  ·  Vũ Gia Khải (B22DCKH065)  ·  Nguyễn Nhật Tân (B22DCKH107)", options: { color: "CADCFC" } },
    ], {
        x: 0.8, y: 5.6, w: 11.5, h: 1.3,
        fontSize: 14, fontFace: "Calibri", lineSpacingMultiple: 1.3,
    });

    slide.addText("Tháng 06/2026", {
        x: 0.8, y: 6.95, w: 4, h: 0.35,
        fontSize: 12, color: MUTED, fontFace: "Calibri",
    });
}

// =====================================================================
// SLIDE 2 - Bối cảnh nghiên cứu
// =====================================================================
{
    const slide = lightSlide("Mở đầu", "Bối cảnh nghiên cứu");

    slide.addText(
        "Nhận dạng biển số xe (License Plate Recognition – LPR) là bài toán quan trọng trong thị giác máy " +
        "tính, ứng dụng trong giao thông thông minh, giám sát an ninh, quản lý bãi đỗ xe và điều tra vi phạm.",
        { x: 0.7, y: 1.7, w: 7.1, h: 1.3, fontSize: 15, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    slide.addText(
        "Trong thực tế, camera giám sát thường đặt xa phương tiện, góc nhìn xiên, ánh sáng thay đổi và chịu " +
        "ảnh hưởng chuyển động → ảnh biển số thu được có độ phân giải rất thấp, mờ, nhiễu hoặc biến dạng.",
        { x: 0.7, y: 3.05, w: 7.1, h: 1.3, fontSize: 15, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    // highlight card - the competition
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.7, y: 4.55, w: 7.1, h: 1.8, rectRadius: 0.08,
        fill: { color: NAVY }, shadow: shadow(),
    });
    slide.addText("ICPR 2026 LRLPR Challenge", {
        x: 1.0, y: 4.75, w: 6.6, h: 0.4, fontSize: 16, bold: true, color: AMBER, fontFace: "Cambria",
    });
    slide.addText(
        "Tập trung vào nhận dạng biển số từ ảnh độ phân giải thấp, với dữ liệu dạng track gồm nhiều frame " +
        "của cùng một biển số — mở ra hướng khai thác thông tin theo thời gian thay vì chỉ dựa trên một ảnh đơn lẻ.",
        { x: 1.0, y: 5.2, w: 6.6, h: 1.05, fontSize: 13.5, color: "E6ECF7", fontFace: "Calibri", lineSpacingMultiple: 1.2 }
    );

    // right column - pipeline mini diagram
    const rx = 8.2, rw = 4.4;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: rx, y: 1.7, w: rw, h: 4.65, rectRadius: 0.08,
        fill: { color: CARD }, shadow: shadow(),
    });
    slide.addText("Các bước trong hệ thống LPR", {
        x: rx + 0.3, y: 1.95, w: rw - 0.6, h: 0.4, fontSize: 15, bold: true, color: TEXT_DARK, fontFace: "Cambria",
    });

    const steps = [
        ["search", "Phát hiện vùng biển số"],
        ["sync", "Căn chỉnh hình ảnh"],
        ["idcard", "Nhận dạng ký tự"],
        ["check", "Hậu xử lý chuỗi biển số"],
    ];
    let sy = 2.55;
    steps.forEach(([icon, label], i) => {
        iconCircle(slide, icon, rx + 0.3, sy, 0.55, "EAF1FB", "slate");
        slide.addText(label, {
            x: rx + 1.0, y: sy, w: rw - 1.3, h: 0.55, fontSize: 14, color: TEXT_DARK, valign: "middle", fontFace: "Calibri",
        });
        if (i < steps.length - 1) {
            slide.addShape(pres.shapes.LINE, {
                x: rx + 0.575, y: sy + 0.55, w: 0, h: 0.4,
                line: { color: "C7D2E3", width: 1.5, dashType: "dash" },
            });
        }
        sy += 0.95;
    });

    pageNumber(slide, 2);
}

// =====================================================================
// SLIDE 3 - Lý do chọn đề tài & Thách thức
// =====================================================================
{
    const slide = lightSlide("Mở đầu", "Lý do chọn đề tài và Thách thức");

    slide.addText(
        "Đề tài LowResolution-LPR có tính ứng dụng thực tế cao và chứa nhiều thách thức kỹ thuật đáng chú ý:",
        { x: 0.7, y: 1.65, w: 11.9, h: 0.45, fontSize: 15, color: TEXT_DARK, fontFace: "Calibri", italic: true }
    );

    const challenges = [
        ["search", "Ảnh kích thước rất nhỏ", "Nhiều biển số chỉ khoảng 30 × 15 pixels, khiến chi tiết ký tự bị mất mát nghiêm trọng."],
        ["random", "Mờ, nhiễu, biến dạng", "Biển số bị mờ, nhiễu, lệch góc hoặc biến dạng do chuyển động và góc chụp camera."],
        ["idcard", "Hai định dạng biển số", "Dữ liệu gồm cả Brazilian và Mercosur, tạo khác biệt về mẫu ký tự cần nhận dạng."],
        ["balance", "Mất cân bằng dữ liệu", "Scenario-B mất cân bằng rõ rệt giữa Brazilian/Mercosur, mô hình dễ thiên lệch."],
        ["eye", "Test blind không nhãn", "Tập test blind chỉ cung cấp ảnh LR, yêu cầu mô hình tổng quát hóa tốt."],
        ["video", "Multi-frame track", "Khai thác thông tin từ nhiều frame liên tiếp thay vì một frame đơn lẻ."],
    ];

    const cols = 3, cardW = 3.95, cardH = 1.9, gx = 0.25, gy = 0.25;
    const startX = 0.7, startY = 2.35;
    challenges.forEach(([icon, title, desc], i) => {
        const col = i % cols, row = Math.floor(i / cols);
        const x = startX + col * (cardW + gx);
        const y = startY + row * (cardH + gy);
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
            x, y, w: cardW, h: cardH, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow(),
        });
        iconCircle(slide, icon, x + 0.25, y + 0.25, 0.55, "FBEFD0", "navy");
        slide.addText(title, {
            x: x + 0.95, y: y + 0.22, w: cardW - 1.2, h: 0.6, fontSize: 14.5, bold: true, color: TEXT_DARK,
            fontFace: "Cambria", valign: "middle",
        });
        slide.addText(desc, {
            x: x + 0.25, y: y + 0.92, w: cardW - 0.5, h: 0.9, fontSize: 12, color: MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.15,
        });
    });

    pageNumber(slide, 3);
}

// =====================================================================
// SLIDE 4 - Mục tiêu nghiên cứu
// =====================================================================
{
    const slide = lightSlide("Mở đầu", "Mục tiêu nghiên cứu");

    const goals = [
        ["route", "Pipeline end-to-end", "Xây dựng pipeline nhận dạng biển số end-to-end: đọc dữ liệu track, tiền xử lý ảnh, huấn luyện mô hình, suy luận và xuất bảng dự đoán."],
        ["sitemap", "Kiến trúc ResTranOCR", "Thiết kế mô hình kết hợp STN, ResNet, Attention Fusion và Transformer Encoder để nhận dạng chuỗi biển số 7 ký tự."],
        ["layers", "Khai thác multi-frame", "Sử dụng cơ chế Attention Fusion, cho phép mô hình tự học frame nào có chất lượng tốt hơn trong cùng một track."],
        ["bar", "Đánh giá toàn diện", "Đánh giá hệ thống qua phân tích dữ liệu, kết quả dự đoán, pattern biển số và các thử nghiệm kiến trúc (ResNet, Transformer, BiLSTM)."],
    ];

    const cardW = 5.9, cardH = 2.3, gx = 0.3, gy = 0.3, startX = 0.7, startY = 1.75;
    goals.forEach(([icon, title, desc], i) => {
        const col = i % 2, row = Math.floor(i / 2);
        const x = startX + col * (cardW + gx);
        const y = startY + row * (cardH + gy);
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
            x, y, w: cardW, h: cardH, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow(),
        });
        slide.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.7, h: 0.7, fill: { color: NAVY } });
        slide.addImage({ path: iconPath(icon, "amber"), x: x + 0.45, y: y + 0.45, w: 0.4, h: 0.4 });
        slide.addText(`Mục tiêu ${i + 1}: ${title}`, {
            x: x + 1.2, y: y + 0.28, w: cardW - 1.4, h: 0.5, fontSize: 16, bold: true, color: TEXT_DARK, fontFace: "Cambria", valign: "middle",
        });
        slide.addText(desc, {
            x: x + 0.3, y: y + 1.05, w: cardW - 0.6, h: 1.1, fontSize: 13, color: MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.25,
        });
    });

    pageNumber(slide, 4);
}

// =====================================================================
// SLIDE 5 - Cơ sở lý thuyết: LPR & OCR
// =====================================================================
{
    const slide = lightSlide("Cơ sở lý thuyết", "Bài toán LPR và OCR");

    slide.addText(
        "License Plate Recognition có thể được xem như một biến thể đặc thù của Optical Character " +
        "Recognition (OCR). Đầu vào là ảnh vùng biển số hoặc một chuỗi frame chứa cùng biển số, đầu ra là chuỗi ký tự biển số.",
        { x: 0.7, y: 1.7, w: 7.0, h: 1.3, fontSize: 15.5, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.3 }
    );

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.7, y: 3.1, w: 7.0, h: 1.6, rectRadius: 0.08, fill: { color: "FFF6DE" }, shadow: shadow(),
    });
    slide.addImage({ path: iconPath("warn", "navy"), x: 1.0, y: 3.35, w: 0.55, h: 0.55 });
    slide.addText(
        "Khác với OCR văn bản thông thường, LPR có không gian ký tự nhỏ hơn nhưng yêu cầu độ chính xác " +
        "tuyệt đối cao: chỉ cần sai một ký tự thì toàn bộ biển số bị xem là nhận dạng sai.",
        { x: 1.75, y: 3.3, w: 5.75, h: 1.2, fontSize: 14, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.25, valign: "middle" }
    );

    slide.addText(
        "Trong đề tài này, chuỗi biển số được chuẩn hóa về độ dài 7 ký tự. Mỗi vị trí trong chuỗi được " +
        "mô hình hóa như một bài toán phân loại đa lớp với 36 lớp.",
        { x: 0.7, y: 4.95, w: 7.0, h: 1.1, fontSize: 14.5, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    // right: 36-class breakdown
    const rx = 8.1, rw = 4.55;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: rx, y: 1.7, w: rw, h: 4.65, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow(),
    });
    slide.addText("Tập 36 ký tự / vị trí", {
        x: rx + 0.3, y: 1.95, w: rw - 0.6, h: 0.4, fontSize: 16, bold: true, color: AMBER, fontFace: "Cambria",
    });

    slide.addText("10", { x: rx + 0.45, y: 2.6, w: 1.8, h: 1.1, fontSize: 48, bold: true, color: "FFFFFF", fontFace: "Cambria", align: "center" });
    slide.addText("chữ số\n0–9", { x: rx + 0.45, y: 3.6, w: 1.8, h: 0.7, fontSize: 13, color: "CADCFC", fontFace: "Calibri", align: "center" });

    slide.addText("+", { x: rx + 2.2, y: 2.85, w: 0.6, h: 1, fontSize: 36, bold: true, color: AMBER, align: "center" });

    slide.addText("26", { x: rx + 2.8, y: 2.6, w: 1.8, h: 1.1, fontSize: 48, bold: true, color: "FFFFFF", fontFace: "Cambria", align: "center" });
    slide.addText("chữ cái\nA–Z", { x: rx + 2.8, y: 3.6, w: 1.8, h: 0.7, fontSize: 13, color: "CADCFC", fontFace: "Calibri", align: "center" });

    slide.addShape(pres.shapes.LINE, { x: rx + 0.35, y: 4.55, w: rw - 0.7, h: 0, line: { color: "3E4E78", width: 1 } });

    slide.addText("= 36 lớp ký tự  ×  7 vị trí", {
        x: rx + 0.3, y: 4.75, w: rw - 0.6, h: 0.5, fontSize: 18, bold: true, color: "FFFFFF", fontFace: "Cambria", align: "center",
    });
    slide.addText("Đầu ra mô hình: logits kích thước (B, 7, 36)", {
        x: rx + 0.3, y: 5.35, w: rw - 0.6, h: 0.9, fontSize: 13, color: "CADCFC", fontFace: "Calibri", align: "center", italic: true, lineSpacingMultiple: 1.2,
    });

    pageNumber(slide, 5);
}

// =====================================================================
// SLIDE 6 - Nhận dạng độ phân giải thấp & các thành phần học sâu
// =====================================================================
{
    const slide = lightSlide("Cơ sở lý thuyết", "Nhận dạng độ phân giải thấp & các thành phần học sâu");

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.7, y: 1.65, w: 12.0, h: 1.25, rectRadius: 0.08, fill: { color: "EAF1FB" }, shadow: shadow(),
    });
    slide.addText(
        "Ảnh độ phân giải thấp làm mất chi tiết nét mảnh, đường cong và khoảng cách giữa ký tự. Các ký tự dễ nhầm: O/0, I/1, B/8, S/5. " +
        "Hướng tiếp cận multi-frame khai thác: một frame có thể bị mờ, nhưng frame khác có thể rõ hơn ở một vài ký tự — cần cơ chế hợp nhất có chọn lọc.",
        { x: 1.0, y: 1.65, w: 11.4, h: 1.25, fontSize: 14.5, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.25, valign: "middle" }
    );

    const comps = [
        ["sitemap", "STN", "Học phép biến đổi hình học để căn chỉnh ảnh đầu vào, giảm ảnh hưởng lệch góc và biến dạng phối cảnh."],
        ["layers", "ResNet", "Backbone trích xuất đặc trưng hình ảnh. Trọng số pretrained ImageNet giúp khởi đầu từ biểu diễn thị giác mạnh."],
        ["random", "Attention Fusion", "Học trọng số quan trọng của từng frame trong cùng track, tổng hợp đặc trưng bằng trung bình có trọng số."],
        ["brain", "Transformer Encoder", "Mô hình hóa quan hệ ngữ cảnh giữa các vị trí ký tự theo chiều ngang biển số bằng self-attention."],
        ["arrows", "BiLSTM", "Thử nghiệm như mô hình chuỗi thay thế Transformer, nhẹ hơn nhưng vẫn nắm bắt phụ thuộc hai chiều."],
    ];

    const cardW = 2.32, cardH = 3.55, gx = 0.18, startX = 0.7, startY = 3.15;
    comps.forEach(([icon, title, desc], i) => {
        const x = startX + i * (cardW + gx);
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
            x, y: startY, w: cardW, h: cardH, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow(),
        });
        iconCircle(slide, icon, x + (cardW - 0.7) / 2, startY + 0.25, 0.7, NAVY, "amber");
        slide.addText(title, {
            x: x + 0.1, y: startY + 1.1, w: cardW - 0.2, h: 0.5, fontSize: 14.5, bold: true, color: TEXT_DARK, fontFace: "Cambria", align: "center",
        });
        slide.addText(desc, {
            x: x + 0.18, y: startY + 1.65, w: cardW - 0.36, h: 1.8, fontSize: 11.5, color: MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.18, align: "left",
        });
    });

    pageNumber(slide, 6);
}

// =====================================================================
// SLIDE 7 - Bài toán & Tập dữ liệu
// =====================================================================
{
    const slide = lightSlide("Dữ liệu", "Mô tả bài toán & Tập dữ liệu ICPR 2026 LRLPR");

    // formula box
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.7, y: 1.65, w: 12.0, h: 1.0, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow(),
    });
    slide.addText([
        { text: "Bài toán:  ", options: { color: AMBER, bold: true } },
        { text: "Với mỗi track gồm F frame LR  X = {x", options: { color: "FFFFFF" } },
        { text: "1", options: { color: "FFFFFF", subscript: true } },
        { text: ", x", options: { color: "FFFFFF" } },
        { text: "2", options: { color: "FFFFFF", subscript: true } },
        { text: ", …, x", options: { color: "FFFFFF" } },
        { text: "F", options: { color: "FFFFFF", subscript: true } },
        { text: "} của cùng một biển số → dự đoán chuỗi 7 ký tự  y = (y", options: { color: "FFFFFF" } },
        { text: "1", options: { color: "FFFFFF", subscript: true } },
        { text: ",…, y", options: { color: "FFFFFF" } },
        { text: "7", options: { color: "FFFFFF", subscript: true } },
        { text: "),  y", options: { color: "FFFFFF" } },
        { text: "i", options: { color: "FFFFFF", italic: true, subscript: true } },
        { text: " ∈ {0..9, A..Z}.  Cấu hình chính dùng 5 frame LR / track.", options: { color: "FFFFFF" } },
    ], { x: 1.0, y: 1.65, w: 11.4, h: 1.0, fontSize: 14.5, fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.2 });

    // table
    const tableRows = [
        [
            { text: "Tập dữ liệu", options: { bold: true, color: "FFFFFF", fill: { color: SLATE } } },
            { text: "Kịch bản", options: { bold: true, color: "FFFFFF", fill: { color: SLATE } } },
            { text: "Loại biển", options: { bold: true, color: "FFFFFF", fill: { color: SLATE } } },
            { text: "Số tracks", options: { bold: true, color: "FFFFFF", fill: { color: SLATE }, align: "right" } },
            { text: "Tổng frames", options: { bold: true, color: "FFFFFF", fill: { color: SLATE }, align: "right" } },
        ],
        ["Train", "Scenario-A", "Brazilian", "4,977", "≈ 49,770"],
        ["Train", "Scenario-A", "Mercosur", "4,960", "≈ 49,600"],
        ["Train", "Scenario-B", "Brazilian", "1,959", "≈ 19,590"],
        ["Train", "Scenario-B", "Mercosur", "7,952", "≈ 79,520"],
        ["Test blind", "–", "Mixed", "1,000", "5,000"],
        ["Test public", "–", "Mixed", "152", "1,520"],
        [
            { text: "Tổng train", options: { bold: true, fill: { color: "EAF1FB" } } },
            { text: "–", options: { fill: { color: "EAF1FB" } } },
            { text: "–", options: { fill: { color: "EAF1FB" } } },
            { text: "19,848", options: { bold: true, align: "right", fill: { color: "EAF1FB" } } },
            { text: "≈ 198,480", options: { bold: true, align: "right", fill: { color: "EAF1FB" } } },
        ],
    ].map(row => row.map(cell => {
        if (typeof cell === "string") {
            return { text: cell, options: { color: TEXT_DARK, align: cell.match(/^[\d≈,]/) ? "right" : "left" } };
        }
        return cell;
    }));

    slide.addTable(tableRows, {
        x: 0.7, y: 2.95, w: 8.5, h: 3.6,
        colW: [1.7, 1.7, 1.7, 1.7, 1.7],
        fontSize: 13, fontFace: "Calibri", border: { pt: 0.75, color: "DCE4F0" },
        rowH: 0.45, valign: "middle",
    });

    // right: project summary card
    const rx = 9.55, rw = 3.1;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: rx, y: 2.95, w: rw, h: 3.6, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow(),
    });
    slide.addText("Tổng quan dữ liệu (dự án)", {
        x: rx + 0.25, y: 3.15, w: rw - 0.5, h: 0.4, fontSize: 14, bold: true, color: TEXT_DARK, fontFace: "Cambria",
    });
    const facts = [
        "19,848 tracks, mỗi track 10 frame",
        "100,000 ảnh LR + 100,000 ảnh HR",
        "100,000 samples, 700,000 ký tự nhãn",
        "Brazilian: ABC1234 (3 chữ + 4 số)",
        "Mercosur: ABC1D23 (vị trí 5 có thể là chữ)",
    ];
    let fy = 3.65;
    facts.forEach(f => {
        slide.addShape(pres.shapes.OVAL, { x: rx + 0.25, y: fy + 0.08, w: 0.12, h: 0.12, fill: { color: AMBER } });
        slide.addText(f, { x: rx + 0.5, y: fy - 0.05, w: rw - 0.75, h: 0.55, fontSize: 12, color: MUTED, fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.1 });
        fy += 0.58;
    });

    pageNumber(slide, 7);
}

// =====================================================================
// SLIDE 8 - EDA: phân bố kích thước ảnh
// =====================================================================
{
    const slide = lightSlide("Dữ liệu — EDA", "Phân tích khám phá dữ liệu: kích thước ảnh");

    slide.addText(
        "Sự chênh lệch kích thước lớn giữa các ảnh cho thấy bước resize về kích thước chuẩn 32 × 128 là cần thiết để huấn luyện theo batch.",
        { x: 0.7, y: 1.6, w: 12.0, h: 0.45, fontSize: 14, italic: true, color: TEXT_DARK, fontFace: "Calibri" }
    );

    // LR card
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.7, y: 2.2, w: 6.0, h: 4.55, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow(),
    });
    iconCircle(slide, "image", 1.0, 2.45, 0.6, "EAF1FB", "slate");
    slide.addText("Ảnh LR (Low-Resolution)", { x: 1.75, y: 2.45, w: 4.6, h: 0.6, fontSize: 16, bold: true, color: TEXT_DARK, fontFace: "Cambria", valign: "middle" });

    slide.addChart(pres.charts.BAR, [
        { name: "Width (px)", labels: ["Min", "Max"], values: [24, 68] },
        { name: "Height (px)", labels: ["Min", "Max"], values: [12, 36] },
    ], {
        x: 0.95, y: 3.15, w: 5.5, h: 3.4, barDir: "col",
        chartColors: [TEAL, AMBER],
        chartArea: { fill: { color: CARD } },
        catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
        valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
        showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT_DARK,
        legendPos: "b", showLegend: true, valAxisMaxVal: 80,
    });

    // HR card
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 6.9, y: 2.2, w: 6.0, h: 4.55, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow(),
    });
    iconCircle(slide, "images", 7.2, 2.45, 0.6, "FBEFD0", "navy");
    slide.addText("Ảnh HR (High-Resolution)", { x: 7.95, y: 2.45, w: 4.6, h: 0.6, fontSize: 16, bold: true, color: TEXT_DARK, fontFace: "Cambria", valign: "middle" });

    slide.addChart(pres.charts.BAR, [
        { name: "Width (px)", labels: ["Min", "Max"], values: [50, 164] },
        { name: "Height (px)", labels: ["Min", "Max"], values: [22, 156] },
    ], {
        x: 7.15, y: 3.15, w: 5.5, h: 3.4, barDir: "col",
        chartColors: [TEAL, AMBER],
        chartArea: { fill: { color: CARD } },
        catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
        valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
        showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT_DARK,
        legendPos: "b", showLegend: true, valAxisMaxVal: 180,
    });

    pageNumber(slide, 8);
}

// =====================================================================
// SLIDE 9 - EDA: tần suất & phân bố ký tự theo vị trí
// =====================================================================
{
    const slide = lightSlide("Dữ liệu — EDA", "Phân bố ký tự và cấu trúc biển số theo vị trí");

    slide.addText(
        "Toàn bộ tập train: 100,000 samples, 700,000 ký tự hợp lệ. Tần suất ký tự không đồng đều: A, B xuất hiện nhiều ở vị trí đầu; " +
        "K, L, M, N, O, U xuất hiện ít hơn — phản ánh đặc trưng định dạng biển số thật và tạo mất cân bằng trong quá trình học.",
        { x: 0.7, y: 1.6, w: 12.0, h: 0.85, fontSize: 14, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    // position structure cards
    const positions = [
        ["1 – 3", "Chữ cái", "Phù hợp cả hai định dạng Brazilian và Mercosur"],
        ["4, 6, 7", "Chữ số", "Tương ứng phần số cuối của biển số"],
        ["5", "Chữ cái hoặc chữ số", "Phân biệt Brazilian (ABC1234) và Mercosur (ABC1D23)"],
    ];

    const cardW = 3.85, cardH = 2.0, gx = 0.25, startX = 0.7, startY = 2.65;
    positions.forEach(([pos, type, desc], i) => {
        const x = startX + i * (cardW + gx);
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
            x, y: startY, w: cardW, h: cardH, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow(),
        });
        slide.addShape(pres.shapes.OVAL, { x: x + 0.25, y: startY + 0.25, w: 0.9, h: 0.9, fill: { color: NAVY } });
        slide.addText(pos, { x: x + 0.25, y: startY + 0.25, w: 0.9, h: 0.9, fontSize: 18, bold: true, color: AMBER, align: "center", valign: "middle", fontFace: "Cambria" });
        slide.addText("Vị trí " + pos, { x: x + 1.3, y: startY + 0.28, w: cardW - 1.5, h: 0.35, fontSize: 12, color: MUTED, fontFace: "Calibri" });
        slide.addText(type, { x: x + 1.3, y: startY + 0.55, w: cardW - 1.5, h: 0.5, fontSize: 16, bold: true, color: TEXT_DARK, fontFace: "Cambria" });
        slide.addText(desc, { x: x + 0.25, y: startY + 1.25, w: cardW - 0.5, h: 0.65, fontSize: 12, color: MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.15 });
    });

    // bottom: example plates
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.7, y: 4.95, w: 12.0, h: 1.8, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow(),
    });
    slide.addText("Ví dụ định dạng biển số", { x: 1.0, y: 5.15, w: 4, h: 0.4, fontSize: 14, bold: true, color: AMBER, fontFace: "Cambria" });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: 5.65, w: 2.6, h: 0.85, rectRadius: 0.08, fill: { color: AMBER } });
    slide.addText("A B L  1 2 3 4", { x: 1.0, y: 5.65, w: 2.6, h: 0.85, fontSize: 19, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Courier New", charSpacing: 2 });
    slide.addText("Brazilian — 3 chữ cái + 4 chữ số", { x: 1.0, y: 6.55, w: 2.6, h: 0.3, fontSize: 11, color: "CADCFC", align: "center", fontFace: "Calibri" });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.1, y: 5.65, w: 2.6, h: 0.85, rectRadius: 0.08, fill: { color: AMBER } });
    slide.addText("A B C  1 D 2 3", { x: 4.1, y: 5.65, w: 2.6, h: 0.85, fontSize: 19, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Courier New", charSpacing: 2 });
    slide.addText("Mercosur — vị trí 5 là chữ cái", { x: 4.1, y: 6.55, w: 2.6, h: 0.3, fontSize: 11, color: "CADCFC", align: "center", fontFace: "Calibri" });

    slide.addText(
        "Scenario-B: Mercosur chiếm khoảng 80.2% — mất cân bằng mạnh giữa hai định dạng, ảnh hưởng tới khả năng học cân bằng của mô hình.",
        { x: 7.2, y: 5.65, w: 5.3, h: 1.0, fontSize: 13, color: "E6ECF7", fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.25, italic: true }
    );

    pageNumber(slide, 9);
}

// =====================================================================
// SLIDE 10 - Tổ chức, tiền xử lý & Synthetic LR
// =====================================================================
{
    const slide = lightSlide("Dữ liệu", "Tổ chức, Tiền xử lý & Tạo mẫu Synthetic LR");

    // left: preprocessing
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.7, y: 1.65, w: 5.9, h: 5.1, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow(),
    });
    iconCircle(slide, "cogs", 1.0, 1.9, 0.6, "EAF1FB", "slate");
    slide.addText("Tổ chức & Tiền xử lý", { x: 1.75, y: 1.9, w: 4.6, h: 0.6, fontSize: 16, bold: true, color: TEXT_DARK, fontFace: "Cambria", valign: "middle" });

    const prepItems = [
        "Mỗi track: thư mục nhiều frame của cùng biển số, metadata gồm plate_text, plate_layout, tọa độ 4 góc.",
        "Scenario-A: biển số crop tốt, Brazilian/Mercosur gần cân bằng.",
        "Scenario-B: từ video camera thực tế, phức tạp hơn, Mercosur ~80.2%.",
        "Resize ảnh về 32 × 128, chuẩn hóa theo ImageNet mean/std.",
        "Gom 5 frame LR thành một mẫu đầu vào.",
        "Vocabulary nhãn: 0–9, A–Z (36 ký tự).",
    ];
    let py = 2.65;
    prepItems.forEach(t => {
        slide.addImage({ path: iconPath("check", "navy"), x: 1.0, y: py + 0.03, w: 0.28, h: 0.28 });
        slide.addText(t, { x: 1.45, y: py - 0.05, w: 5.0, h: 0.65, fontSize: 12.5, color: TEXT_DARK, fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.1 });
        py += 0.68;
    });

    // right: synthetic LR
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 6.85, y: 1.65, w: 5.8, h: 5.1, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow(),
    });
    iconCircle(slide, "flask", 7.15, 1.9, 0.6, AMBER, "navy");
    slide.addText("Synthetic LR từ ảnh HR (HR-degrade)", { x: 7.95, y: 1.9, w: 4.6, h: 0.6, fontSize: 16, bold: true, color: AMBER, fontFace: "Cambria", valign: "middle" });

    slide.addText(
        "Ảnh HR được làm giảm chất lượng bằng các phép biến đổi: blur, motion blur, nhiễu, nén ảnh và downscale — " +
        "tạo thêm các biến thể synthetic LR có nhãn giống track gốc, tăng độ đa dạng dữ liệu.",
        { x: 7.15, y: 2.65, w: 5.2, h: 1.1, fontSize: 13, color: "E6ECF7", fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    // pipeline visualization: HR -> degrade -> Synthetic LR
    const py2 = 4.0;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.15, y: py2, w: 1.6, h: 1.0, rectRadius: 0.08, fill: { color: "FFFFFF" } });
    slide.addText("Ảnh HR\nrõ nét", { x: 7.15, y: py2, w: 1.6, h: 1.0, fontSize: 12, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Calibri" });

    slide.addShape(pres.shapes.RECTANGLE, { x: 8.85, y: py2 + 0.42, w: 0.7, h: 0, line: { color: AMBER, width: 2, endArrowType: "triangle" } });
    slide.addText("blur · noise\ncompress\ndownscale", { x: 8.7, y: py2 - 0.55, w: 1.0, h: 0.5, fontSize: 9.5, color: "CADCFC", align: "center", fontFace: "Calibri" });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.65, y: py2, w: 1.65, h: 1.0, rectRadius: 0.08, fill: { color: SLATE } });
    slide.addText("Synthetic\nLR", { x: 9.65, y: py2, w: 1.65, h: 1.0, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", fontFace: "Calibri" });

    slide.addShape(pres.shapes.RECTANGLE, { x: 11.4, y: py2 + 0.42, w: 0.7, h: 0, line: { color: AMBER, width: 2, endArrowType: "triangle" } });

    slide.addText("nhãn track gốc\n(giữ nguyên)", { x: 11.25, y: py2 - 0.55, w: 1.1, h: 0.5, fontSize: 9.5, color: "CADCFC", align: "center", fontFace: "Calibri" });

    slide.addText(
        "Train: mỗi track đóng góp mẫu LR gốc + mẫu synthetic LR.\nValidation/Test: chỉ giữ dữ liệu LR thực để đánh giá đúng điều kiện nhận dạng.",
        { x: 7.15, y: 5.3, w: 5.2, h: 1.3, fontSize: 13, color: "CADCFC", fontFace: "Calibri", lineSpacingMultiple: 1.3, italic: true }
    );

    pageNumber(slide, 10);
}

// =====================================================================
// SLIDE 11 - Kiến trúc ResTranOCR tổng quan
// =====================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: NAVY };

    slide.addText("PHƯƠNG PHÁP", { x: 0.7, y: 0.4, w: 10, h: 0.35, fontSize: 13, color: AMBER, bold: true, fontFace: "Calibri", charSpacing: 2 });
    slide.addText("Kiến trúc tổng thể ResTranOCR", { x: 0.7, y: 0.7, w: 12, h: 0.8, fontSize: 30, color: "FFFFFF", bold: true, fontFace: "Cambria" });

    // pipeline boxes
    const boxes = [
        ["5 Frame LR\n32×128", "EAF1FB", TEXT_DARK],
        ["STN Block\n(affine align)", AMBER, NAVY],
        ["ResNet-50\nFeature Extractor", "3E7CB1", "FFFFFF"],
        ["Attention\nFusion", AMBER, NAVY],
        ["Transformer\nEncoder ×3", "3E7CB1", "FFFFFF"],
        ["Linear Head\n(B,7,36)", "EAF1FB", TEXT_DARK],
        ["AVL5215", "FFFFFF", NAVY],
    ];
    const bw = 1.62, bh = 1.5, gap = 0.155, startX = 0.7, by = 2.35;
    boxes.forEach((b, i) => {
        const x = startX + i * (bw + gap);
        if (i === 6) {
            slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w: bw, h: bh, rectRadius: 0.1, fill: { color: AMBER }, line: { color: "FFFFFF", width: 1.5 } });
            slide.addText(b[0], { x, y: by, w: bw, h: bh, fontSize: 17, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Courier New", charSpacing: 1 });
        } else {
            slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w: bw, h: bh, rectRadius: 0.1, fill: { color: b[1] } });
            slide.addText(b[0], { x: x + 0.06, y: by, w: bw - 0.12, h: bh, fontSize: 12.5, bold: true, color: b[2], align: "center", valign: "middle", fontFace: "Calibri", lineSpacingMultiple: 1.1 });
        }
        if (i < boxes.length - 1) {
            slide.addShape(pres.shapes.RECTANGLE, {
                x: x + bw, y: by + bh / 2, w: gap, h: 0,
                line: { color: AMBER, width: 2, endArrowType: "triangle" },
            });
        }
    });

    // step descriptions below
    const steps = [
        "1 — Đầu vào (B,5,3,32,128), reshape thành (B×5,3,32,128) xử lý từng frame.",
        "2 — STNBlock dự đoán ma trận affine 2×3, căn chỉnh ảnh bằng affine_grid + grid_sample.",
        "3 — ResNet-50 pretrained, stride layer3/4 chỉnh (2,2)→(2,1) giữ thông tin chiều ngang; chiếu 2048→512 kênh, pooling còn (B×5,512,1,16).",
        "4 — AttentionFusion hợp nhất 5 đặc trưng → (B,512,1,16) cho mỗi track.",
        "5 — Chuyển thành chuỗi dài 16 + positional encoding → Transformer Encoder (3 lớp, 8 heads).",
        "6 — AdaptiveAvgPool1d: 16 → 7 timestep; Linear Head dự đoán 36 lớp/timestep → logits (B,7,36).",
    ];
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 4.25, w: 12.0, h: 2.85, rectRadius: 0.08, fill: { color: SLATE } });
    slide.addText("Luồng xử lý", { x: 1.0, y: 4.45, w: 4, h: 0.4, fontSize: 14, bold: true, color: AMBER, fontFace: "Cambria" });
    slide.addText(steps.map((s, i) => ({ text: s, options: { breakLine: i < steps.length - 1, bullet: false } })), {
        x: 1.0, y: 4.85, w: 11.4, h: 2.15, fontSize: 13, color: "E6ECF7", fontFace: "Calibri", lineSpacingMultiple: 1.35,
    });

    pageNumber(slide, 11);
}

// =====================================================================
// SLIDE 12 - STN
// =====================================================================
{
    const slide = lightSlide("Phương pháp 4.2", "Spatial Transformer Network (STN)");

    slide.addText(
        "STN được đặt ở đầu pipeline để giảm ảnh hưởng của lệch góc, dịch chuyển và biến dạng phối cảnh trước khi trích xuất đặc trưng bằng ResNet.",
        { x: 0.7, y: 1.65, w: 7.1, h: 0.75, fontSize: 14.5, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    const stnSteps = [
        ["search", "Localization network", "Các lớp convolution, pooling và fully-connected dự đoán 6 tham số của ma trận affine 2×3."],
        ["sync", "Khởi tạo gần identity", "Lớp cuối được khởi tạo gần phép biến đổi đồng nhất — mô hình bắt đầu từ ảnh gốc rồi dần học cách căn chỉnh."],
        ["target", "affine_grid + grid_sample", "Sinh sampling grid và lấy mẫu lại ảnh bằng nội suy, đưa vùng biển số về trạng thái thẳng, ổn định hơn."],
    ];
    let sy = 2.6;
    stnSteps.forEach(([icon, title, desc], i) => {
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: sy, w: 7.1, h: 1.25, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
        iconCircle(slide, icon, 0.95, sy + 0.25, 0.7, NAVY, "amber");
        slide.addText(`${i + 1}. ${title}`, { x: 1.85, y: sy + 0.15, w: 5.85, h: 0.4, fontSize: 14, bold: true, color: TEXT_DARK, fontFace: "Cambria" });
        slide.addText(desc, { x: 1.85, y: sy + 0.55, w: 5.85, h: 0.65, fontSize: 12, color: MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.15 });
        sy += 1.4;
    });

    // right: before/after illustration using shapes (plate)
    const rx = 8.1, rw = 4.55;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.65, w: rw, h: 5.1, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    slide.addText("Trước và sau STN", { x: rx + 0.3, y: 1.9, w: rw - 0.6, h: 0.4, fontSize: 16, bold: true, color: AMBER, fontFace: "Cambria", align: "center" });

    // before - skewed plate
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: rx + 0.65, y: 2.6, w: 3.25, h: 1.05, rectRadius: 0.06, fill: { color: "DDE6F4" }, rotate: -8,
    });
    slide.addText("KAS 4B97", { x: rx + 0.65, y: 2.6, w: 3.25, h: 1.05, fontSize: 18, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Courier New", rotate: -8, charSpacing: 2 });
    slide.addText("Ảnh gốc (lệch góc)", { x: rx + 0.3, y: 3.85, w: rw - 0.6, h: 0.35, fontSize: 12, color: "CADCFC", align: "center", italic: true, fontFace: "Calibri" });

    // arrow down
    slide.addShape(pres.shapes.RECTANGLE, { x: rx + rw / 2, y: 4.25, w: 0, h: 0.45, line: { color: AMBER, width: 2.5, endArrowType: "triangle" } });

    // after - straight plate
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.65, y: 4.85, w: 3.25, h: 1.05, rectRadius: 0.06, fill: { color: AMBER } });
    slide.addText("KAS 4B97", { x: rx + 0.65, y: 4.85, w: 3.25, h: 1.05, fontSize: 18, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Courier New", charSpacing: 2 });
    slide.addText("Sau STN — căn chỉnh thẳng", { x: rx + 0.3, y: 6.05, w: rw - 0.6, h: 0.35, fontSize: 12, color: "CADCFC", align: "center", italic: true, fontFace: "Calibri" });

    pageNumber(slide, 12);
}

// =====================================================================
// SLIDE 13 - Multi-frame Attention Fusion
// =====================================================================
{
    const slide = lightSlide("Phương pháp 4.3", "Multi-frame Attention Fusion");

    slide.addText(
        "Chất lượng các frame trong cùng track không đồng đều. Mạng score_net (hai convolution 1×1) sinh điểm chất lượng cho từng frame tại từng vị trí không gian, " +
        "sau đó chuẩn hóa softmax theo chiều frame:",
        { x: 0.7, y: 1.65, w: 7.1, h: 1.1, fontSize: 14.5, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    // formula box
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 2.85, w: 7.1, h: 1.4, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    slide.addText([
        { text: "α", options: { italic: true, color: AMBER, fontSize: 22 } },
        { text: "i", options: { italic: true, color: AMBER, fontSize: 14, subscript: true } },
        { text: "  =  exp(s", options: { color: "FFFFFF", fontSize: 22 } },
        { text: "i", options: { color: "FFFFFF", fontSize: 14, subscript: true } },
        { text: ") / Σ", options: { color: "FFFFFF", fontSize: 22 } },
        { text: "j=1..F", options: { color: "FFFFFF", fontSize: 12, subscript: true } },
        { text: " exp(s", options: { color: "FFFFFF", fontSize: 22 } },
        { text: "j", options: { color: "FFFFFF", fontSize: 14, subscript: true } },
        { text: ")        z = Σ", options: { color: "FFFFFF", fontSize: 22 } },
        { text: "i=1..F", options: { color: "FFFFFF", fontSize: 12, subscript: true } },
        { text: " α", options: { color: "FFFFFF", fontSize: 22, italic: true } },
        { text: "i", options: { color: "FFFFFF", fontSize: 14, subscript: true } },
        { text: " h", options: { color: "FFFFFF", fontSize: 22, italic: true } },
        { text: "i", options: { color: "FFFFFF", fontSize: 14, subscript: true } },
    ], { x: 1.0, y: 2.85, w: 6.5, h: 0.8, fontFace: "Cambria", valign: "middle" });
    slide.addText("hᵢ: đặc trưng frame i   ·   sᵢ: điểm chất lượng học được   ·   z: đặc trưng sau hợp nhất", {
        x: 1.0, y: 3.65, w: 6.5, h: 0.5, fontSize: 12, italic: true, color: "CADCFC", fontFace: "Calibri", valign: "middle",
    });

    slide.addText(
        "Khác với lấy trung bình đơn giản, attention fusion cho phép trọng số thay đổi theo từng vị trí trên feature map: " +
        "một frame có thể rõ ở ký tự đầu nhưng mờ ở ký tự cuối, frame khác bổ sung thông tin ở vùng còn lại — fusion thực hiện ở mức đặc trưng, không chọn một frame duy nhất.",
        { x: 0.7, y: 4.45, w: 7.1, h: 1.5, fontSize: 13.5, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.3 }
    );

    // right: visual diagram of fusion
    const rx = 8.1, rw = 4.55;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.65, w: rw, h: 5.1, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
    slide.addText("Sơ đồ hợp nhất 5 frame", { x: rx + 0.3, y: 1.9, w: rw - 0.6, h: 0.4, fontSize: 15, bold: true, color: TEXT_DARK, fontFace: "Cambria", align: "center" });

    // 5 frame boxes on the left of card with weights
    const weights = ["α₁ = 0.31", "α₂ = 0.12", "α₃ = 0.27", "α₄ = 0.08", "α₅ = 0.22"];
    let fy = 2.55;
    weights.forEach((w, i) => {
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.35, y: fy, w: 1.5, h: 0.55, rectRadius: 0.06, fill: { color: "EAF1FB" } });
        slide.addText(`Frame ${i + 1}`, { x: rx + 0.35, y: fy, w: 1.5, h: 0.55, fontSize: 11, bold: true, color: TEXT_DARK, align: "center", valign: "middle", fontFace: "Calibri" });
        slide.addText(w, { x: rx + 2.0, y: fy, w: 1.0, h: 0.55, fontSize: 12, color: TEAL, align: "left", valign: "middle", fontFace: "Calibri", italic: true });
        slide.addShape(pres.shapes.LINE, { x: rx + 1.85, y: fy + 0.275, w: 1.55, h: (2.55 + 2.6 * 2 + 0.275) - (fy + 0.275), line: { color: "C7D2E3", width: 1 } });
        fy += 0.65;
    });

    // arrow to fused box
    slide.addShape(pres.shapes.RECTANGLE, { x: rx + 3.0, y: 2.55 + 2.6, w: 0.55, h: 0, line: { color: AMBER, width: 2.5, endArrowType: "triangle" } });
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.35, y: 6.0, w: 3.85, h: 0.6, rectRadius: 0.06, fill: { color: AMBER } });
    slide.addText("Fused Feature Map  (B,512,1,16)", { x: rx + 0.35, y: 6.0, w: 3.85, h: 0.6, fontSize: 13, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Calibri" });

    pageNumber(slide, 13);
}

// =====================================================================
// SLIDE 14 - Sequence Modeling & dự đoán ký tự
// =====================================================================
{
    const slide = lightSlide("Phương pháp 4.4", "Sequence Modeling & Dự đoán ký tự");

    slide.addText(
        "Sau fusion, đặc trưng có chiều cao = 1 và chiều rộng = 16 — được xem như chuỗi thị giác theo trục ngang của biển số.",
        { x: 0.7, y: 1.65, w: 12.0, h: 0.5, fontSize: 15, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.2 }
    );

    // sequence flow diagram
    const seqBoxes = [
        ["Feature map\n(B,512,1,16)", "EAF1FB", TEXT_DARK],
        ["+ Positional\nEncoding", AMBER, NAVY],
        ["Transformer\nEncoder", "3E7CB1", "FFFFFF"],
        ["AdaptiveAvgPool1d\n16 → 7", AMBER, NAVY],
        ["Linear Head\n36 lớp / timestep", "EAF1FB", TEXT_DARK],
        ["logits\n(B,7,36)", NAVY, "FFFFFF"],
    ];
    const bw = 1.95, bh = 1.4, gap = 0.18, startX = 0.7, by = 2.4;
    seqBoxes.forEach((b, i) => {
        const x = startX + i * (bw + gap);
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w: bw, h: bh, rectRadius: 0.1, fill: { color: b[1] }, shadow: shadow() });
        slide.addText(b[0], { x: x + 0.05, y: by, w: bw - 0.1, h: bh, fontSize: 13, bold: true, color: b[2], align: "center", valign: "middle", fontFace: "Calibri", lineSpacingMultiple: 1.1 });
        if (i < seqBoxes.length - 1) {
            slide.addShape(pres.shapes.RECTANGLE, { x: x + bw, y: by + bh / 2, w: gap, h: 0, line: { color: TEAL, width: 2, endArrowType: "triangle" } });
        }
    });

    // explanation cards
    const explains = [
        ["arrows", "Positional encoding", "Cộng vào chuỗi để mô hình phân biệt thứ tự trái–phải của các vùng ký tự."],
        ["brain", "Transformer Encoder", "3 lớp, 8 attention heads — mô hình hóa quan hệ ngữ cảnh giữa các vùng ký tự."],
        ["random", "Pooling 16 → 7", "AdaptiveAvgPool1d nén chuỗi 16 bước về đúng 7 bước, tương ứng độ dài nhãn biển số đã chuẩn hóa."],
        ["idcard", "Linear Head", "Mỗi timestep được phân loại thành 1 trong 36 ký tự — 7 bài toán phân loại chia sẻ đặc trưng, vẫn tận dụng ngữ cảnh chuỗi."],
    ];
    const cardW = 2.95, cardH = 2.7, gx2 = 0.2, startX2 = 0.7, startY2 = 4.15;
    explains.forEach(([icon, title, desc], i) => {
        const x = startX2 + i * (cardW + gx2);
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: startY2, w: cardW, h: cardH, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
        iconCircle(slide, icon, x + 0.25, startY2 + 0.25, 0.6, NAVY, "amber");
        slide.addText(title, { x: x + 0.25, y: startY2 + 1.0, w: cardW - 0.5, h: 0.5, fontSize: 14, bold: true, color: TEXT_DARK, fontFace: "Cambria" });
        slide.addText(desc, { x: x + 0.25, y: startY2 + 1.5, w: cardW - 0.5, h: 1.1, fontSize: 12, color: MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.2 });
    });

    pageNumber(slide, 14);
}

// =====================================================================
// SLIDE 15 - Huấn luyện & Suy luận
// =====================================================================
{
    const slide = lightSlide("Phương pháp 4.5", "Huấn luyện & Suy luận");

    // left: training strategy timeline
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.65, w: 6.4, h: 5.1, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
    iconCircle(slide, "rocket", 1.0, 1.9, 0.6, "EAF1FB", "slate");
    slide.addText("Chiến lược Warmup", { x: 1.75, y: 1.9, w: 5, h: 0.6, fontSize: 16, bold: true, color: TEXT_DARK, fontFace: "Cambria", valign: "middle" });

    // timeline steps
    const tlSteps = [
        ["Epoch đầu", "Freeze FeatureExtractor — mô hình học alignment (STN), fusion và sequence modeling trước."],
        ["Sau warmup", "Unfreeze toàn bộ để fine-tune backbone (ResNet-50) cùng các thành phần còn lại."],
        ["Mục đích", "Hạn chế gradient ban đầu phá vỡ trọng số pretrained của ResNet."],
    ];
    let ty = 2.75;
    tlSteps.forEach(([t, d], i) => {
        slide.addShape(pres.shapes.OVAL, { x: 1.0, y: ty, w: 0.35, h: 0.35, fill: { color: AMBER } });
        slide.addText(String(i + 1), { x: 1.0, y: ty, w: 0.35, h: 0.35, fontSize: 13, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Calibri" });
        if (i < tlSteps.length - 1) {
            slide.addShape(pres.shapes.LINE, { x: 1.175, y: ty + 0.35, w: 0, h: 0.85, line: { color: "C7D2E3", width: 1.5, dashType: "dash" } });
        }
        slide.addText(t, { x: 1.55, y: ty - 0.05, w: 2.0, h: 0.45, fontSize: 13, bold: true, color: TEAL, fontFace: "Cambria" });
        slide.addText(d, { x: 1.55, y: ty + 0.35, w: 5.25, h: 0.8, fontSize: 12.5, color: MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.2 });
        ty += 1.25;
    });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: ty + 0.1, w: 5.8, h: 1.1, rectRadius: 0.08, fill: { color: "EAF1FB" } });
    slide.addText([
        { text: "Loss: ", options: { bold: true, color: TEXT_DARK } },
        { text: "CrossEntropyLoss theo từng vị trí ký tự. Đầu ra logits (B,7,36), nhãn (B,7), loss tính sau khi hoán vị tensor về dạng phù hợp phân loại đa lớp.", options: { color: MUTED } },
    ], { x: 1.25, y: ty + 0.1, w: 5.3, h: 1.1, fontSize: 12.5, fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.2 });

    // right: inference
    const rx = 7.3, rw = 5.35;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.65, w: rw, h: 5.1, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    iconCircle(slide, "search", rx + 0.3, 1.9, 0.6, AMBER, "navy");
    slide.addText("Suy luận (Inference)", { x: rx + 1.05, y: 1.9, w: rw - 1.3, h: 0.6, fontSize: 16, bold: true, color: AMBER, fontFace: "Cambria", valign: "middle" });

    const infItems = [
        "Sử dụng trọng số tốt nhất theo validation accuracy.",
        "Dự đoán chuỗi ký tự cho từng track test.",
        "Xuất bảng kết quả: track_id, plate_text.",
        "Hậu xử lý theo pattern biển số để giảm nhầm lẫn ký tự gần giống nhau:",
    ];
    let iy = 2.65;
    infItems.forEach(t => {
        slide.addShape(pres.shapes.OVAL, { x: rx + 0.3, y: iy + 0.08, w: 0.12, h: 0.12, fill: { color: AMBER } });
        slide.addText(t, { x: rx + 0.55, y: iy - 0.05, w: rw - 0.8, h: 0.55, fontSize: 13, color: "E6ECF7", fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.15 });
        iy += 0.6;
    });

    // confusable chars
    const pairs = ["O / 0", "I / 1", "B / 8", "S / 5"];
    let px = rx + 0.55;
    pairs.forEach(p => {
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px, y: iy + 0.15, w: 1.05, h: 0.6, rectRadius: 0.06, fill: { color: SLATE } });
        slide.addText(p, { x: px, y: iy + 0.15, w: 1.05, h: 0.6, fontSize: 15, bold: true, color: AMBER, align: "center", valign: "middle", fontFace: "Courier New" });
        px += 1.2;
    });

    slide.addText(
        "Metric chính: Exact Match Accuracy — một mẫu chỉ được tính đúng nếu cả 7 ký tự đều được dự đoán chính xác theo đúng thứ tự.",
        { x: rx + 0.3, y: 5.55, w: rw - 0.6, h: 1.0, fontSize: 13, color: "CADCFC", fontFace: "Calibri", italic: true, lineSpacingMultiple: 1.25, valign: "middle" }
    );

    pageNumber(slide, 15);
}

// =====================================================================
// SLIDE 16 - Thiết kế thực nghiệm: Môi trường & Hyperparameters
// =====================================================================
{
    const slide = lightSlide("Thiết kế thực nghiệm", "Môi trường tính toán & Siêu tham số");

    // left: environment
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.65, w: 4.4, h: 5.1, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    iconCircle(slide, "cogs", 1.0, 1.9, 0.6, AMBER, "navy");
    slide.addText("Môi trường tính toán", { x: 1.75, y: 1.9, w: 3.2, h: 0.6, fontSize: 15, bold: true, color: AMBER, fontFace: "Cambria", valign: "middle" });

    const envItems = [
        "Python / PyTorch, huấn luyện trên GPU (CUDA)",
        "Pipeline: đọc track → batch multi-frame → resize/augment → train → eval (exact-match) → inference",
        "Chia nội bộ: train / validation / test để theo dõi tổng quát hóa trước khi dự đoán blind test",
    ];
    let ey = 2.7;
    envItems.forEach(t => {
        slide.addShape(pres.shapes.OVAL, { x: 1.0, y: ey + 0.08, w: 0.12, h: 0.12, fill: { color: AMBER } });
        slide.addText(t, { x: 1.25, y: ey - 0.1, w: 3.65, h: 1.1, fontSize: 13, color: "E6ECF7", fontFace: "Calibri", lineSpacingMultiple: 1.2 });
        ey += 1.1;
    });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: ey + 0.2, w: 3.7, h: 1.6, rectRadius: 0.08, fill: { color: SLATE } });
    slide.addText("Internal split", { x: 1.2, y: ey + 0.35, w: 3.3, h: 0.35, fontSize: 12, bold: true, color: AMBER, fontFace: "Cambria" });
    slide.addText("16,000 train · 2,000 validation · 2,000 test", { x: 1.2, y: ey + 0.72, w: 3.3, h: 0.4, fontSize: 13, color: "FFFFFF", fontFace: "Calibri" });
    slide.addText("(+HR-degrade → 32,000 train samples)", { x: 1.2, y: ey + 1.12, w: 3.3, h: 0.4, fontSize: 11.5, color: "CADCFC", italic: true, fontFace: "Calibri" });

    // right: hyperparameters table
    const hpRows = [
        [{ text: "Tham số", options: { bold: true, color: "FFFFFF", fill: { color: SLATE } } }, { text: "Giá trị", options: { bold: true, color: "FFFFFF", fill: { color: SLATE } } }],
        ["Backbone chính", "ResNet-50 (pretrained ImageNet)"],
        ["Embedding dimension", "512"],
        ["Feed-forward dimension", "2048"],
        ["Transformer layers", "3"],
        ["Attention heads", "8"],
        ["Dropout", "0.1"],
        ["Số frame đầu vào", "5"],
        ["Kích thước ảnh", "32 × 128"],
        ["Độ dài nhãn / Số lớp ký tự", "7 ký tự / 36 lớp"],
        ["Optimizer", "Adam, lr = 5 × 10⁻⁴"],
        ["Scheduler", "CosineAnnealingLR"],
        ["Batch size", "64"],
        ["Epochs tối đa", "50"],
        ["Gradient clipping", "max norm = 5.0"],
        ["Early stopping", "10 epoch không cải thiện val acc"],
    ].map((row, i) => i === 0 ? row : row.map((c, ci) => ({ text: c, options: { color: TEXT_DARK, bold: ci === 0, fill: { color: i % 2 === 0 ? "FFFFFF" : "F4F6FA" } } })));

    slide.addTable(hpRows, {
        x: 5.35, y: 1.65, w: 7.3, h: 5.1,
        colW: [3.4, 3.9],
        fontSize: 11.5, fontFace: "Calibri", border: { pt: 0.5, color: "DCE4F0" },
        rowH: 0.32, valign: "middle",
    });

    pageNumber(slide, 16);
}

// =====================================================================
// SLIDE 17 - Các biến thể thử nghiệm & Độ đo đánh giá
// =====================================================================
{
    const slide = lightSlide("Thiết kế thực nghiệm", "Các biến thể thử nghiệm & Độ đo đánh giá");

    // variants table
    const varRows = [
        [{ text: "Biến thể", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } }, { text: "Backbone", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } }, { text: "Sequence model / Ý tưởng", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } }],
        [{ text: "Mô hình chính", options: { bold: true, color: TEXT_DARK, fill: { color: "FFF6DE" } } }, { text: "ResNet-50", options: { color: TEXT_DARK, fill: { color: "FFF6DE" } } }, { text: "Transformer multi-frame với Attention Fusion", options: { color: TEXT_DARK, fill: { color: "FFF6DE" } } }],
        ["Backbone nhẹ hơn", "ResNet-34", "Transformer OCR"],
        ["Decoder thay thế", "ResNet-34", "BiLSTM OCR"],
    ].map((row, i) => i < 2 ? row : row.map((c) => ({ text: c, options: { color: TEXT_DARK, fill: { color: "FFFFFF" } } })));

    slide.addText("Các hướng kiến trúc thử nghiệm", { x: 0.7, y: 1.6, w: 7, h: 0.4, fontSize: 16, bold: true, color: TEXT_DARK, fontFace: "Cambria" });
    slide.addTable(varRows, {
        x: 0.7, y: 2.1, w: 7.4, h: 1.8,
        colW: [2.2, 1.7, 3.5],
        fontSize: 13, fontFace: "Calibri", border: { pt: 0.5, color: "DCE4F0" },
        rowH: 0.45, valign: "middle",
    });

    slide.addText(
        "Mô hình ResNet-50 + Transformer được dùng làm cấu hình chính vì backbone sâu hơn cung cấp đặc trưng thị giác mạnh, " +
        "trong khi Transformer Encoder phù hợp với chuỗi ký tự theo chiều ngang của biển số. ResNet-34 và BiLSTM được dùng để " +
        "đánh giá đánh đổi giữa độ phức tạp mô hình và khả năng nhận dạng.",
        { x: 0.7, y: 4.1, w: 7.4, h: 1.4, fontSize: 13, color: MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.3 }
    );

    // right: Exact match accuracy explainer
    const rx = 8.45, rw = 4.2;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.6, w: rw, h: 5.15, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    slide.addText("Exact Match Accuracy", { x: rx + 0.3, y: 1.85, w: rw - 0.6, h: 0.4, fontSize: 15, bold: true, color: AMBER, fontFace: "Cambria" });
    slide.addText(
        "Một mẫu chỉ được tính đúng nếu cả 7 ký tự đều trùng khớp nhãn thật theo đúng thứ tự.",
        { x: rx + 0.3, y: 2.3, w: rw - 0.6, h: 0.65, fontSize: 12.5, color: "E6ECF7", fontFace: "Calibri", lineSpacingMultiple: 1.2 }
    );

    // example correct
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.3, y: 3.05, w: rw - 0.6, h: 0.6, rectRadius: 0.06, fill: { color: "2E7D32" } });
    slide.addText("Nhãn: ABC1234   Dự đoán: ABC1234  ✓", { x: rx + 0.3, y: 3.05, w: rw - 0.6, h: 0.6, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", fontFace: "Courier New" });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.3, y: 3.8, w: rw - 0.6, h: 0.6, rectRadius: 0.06, fill: { color: "B3261E" } });
    slide.addText("Nhãn: ABC1234   Dự đoán: ABC1235  ✗", { x: rx + 0.3, y: 3.8, w: rw - 0.6, h: 0.6, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", fontFace: "Courier New" });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.3, y: 4.55, w: rw - 0.6, h: 0.6, rectRadius: 0.06, fill: { color: "B3261E" } });
    slide.addText("Nhãn: ABC1234   Dự đoán: ABC1243  ✗", { x: rx + 0.3, y: 4.55, w: rw - 0.6, h: 0.6, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", fontFace: "Courier New" });

    slide.addText(
        "Sai 1 ký tự hoặc sai vị trí ký tự → tính sai toàn bộ mẫu. Metric nghiêm ngặt nhưng phù hợp LPR vì biển số chỉ hữu ích khi toàn bộ chuỗi đúng.",
        { x: rx + 0.3, y: 5.3, w: rw - 0.6, h: 1.25, fontSize: 12.5, italic: true, color: "CADCFC", fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    pageNumber(slide, 17);
}

// =====================================================================
// SLIDE 18 - Kết quả định lượng
// =====================================================================
{
    const slide = lightSlide("Kết quả đánh giá", "Phân tích định lượng (Quantitative Analysis)");

    // big stat
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.7, w: 4.3, h: 5.0, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    slide.addText("Validation Exact-Match\nAccuracy tốt nhất", { x: 1.0, y: 1.95, w: 3.7, h: 0.85, fontSize: 15, bold: true, color: AMBER, fontFace: "Cambria", lineSpacingMultiple: 1.15 });
    slide.addText("0.7437", { x: 1.0, y: 2.85, w: 3.7, h: 1.4, fontSize: 64, bold: true, color: "FFFFFF", align: "center", fontFace: "Cambria" });
    slide.addText("đạt được tại epoch 28", { x: 1.0, y: 4.3, w: 3.7, h: 0.4, fontSize: 13, italic: true, color: "CADCFC", align: "center", fontFace: "Calibri" });

    slide.addShape(pres.shapes.LINE, { x: 1.0, y: 4.95, w: 3.7, h: 0, line: { color: "3E4E78", width: 1 } });

    slide.addText(
        "Cấu hình: ResNet-50 + Transformer\n(mô hình chính)",
        { x: 1.0, y: 5.1, w: 3.7, h: 0.6, fontSize: 13, color: "E6ECF7", align: "center", fontFace: "Calibri", lineSpacingMultiple: 1.2 }
    );
    slide.addText(
        "Tập blind test không công bố nhãn → báo cáo không tự suy diễn accuracy cuối cùng trên blind test.",
        { x: 1.0, y: 5.85, w: 3.7, h: 0.75, fontSize: 11.5, italic: true, color: "CADCFC", align: "center", fontFace: "Calibri", lineSpacingMultiple: 1.2 }
    );

    // middle: 2 supported patterns
    slide.addText("Hai pattern biển số được sinh ra", { x: 5.3, y: 1.7, w: 7.3, h: 0.4, fontSize: 16, bold: true, color: TEXT_DARK, fontFace: "Cambria" });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.3, y: 2.25, w: 7.3, h: 1.5, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
    slide.addText("Brazilian pattern", { x: 5.6, y: 2.45, w: 3, h: 0.4, fontSize: 14, bold: true, color: TEAL, fontFace: "Cambria" });
    slide.addText("[A-Z]{3}[0-9]{4}", { x: 5.6, y: 2.85, w: 3, h: 0.4, fontSize: 14, color: MUTED, fontFace: "Courier New" });
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.4, y: 2.55, w: 2.9, h: 0.9, rectRadius: 0.08, fill: { color: AMBER } });
    slide.addText("AVL 5215", { x: 9.4, y: 2.55, w: 2.9, h: 0.9, fontSize: 22, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Courier New", charSpacing: 3 });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.3, y: 3.9, w: 7.3, h: 1.5, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
    slide.addText("Mercosur pattern", { x: 5.6, y: 4.1, w: 3, h: 0.4, fontSize: 14, bold: true, color: TEAL, fontFace: "Cambria" });
    slide.addText("[A-Z]{3}[0-9][A-Z][0-9]{2}", { x: 5.6, y: 4.5, w: 3, h: 0.4, fontSize: 14, color: MUTED, fontFace: "Courier New" });
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.4, y: 4.2, w: 2.9, h: 0.9, rectRadius: 0.08, fill: { color: AMBER } });
    slide.addText("SFG 1D23", { x: 9.4, y: 4.2, w: 2.9, h: 0.9, fontSize: 22, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Courier New", charSpacing: 3 });

    slide.addText(
        "Sự xuất hiện của cả hai pattern cho thấy mô hình không chỉ học một định dạng cố định mà có khả năng phân biệt các cấu trúc biển số khác nhau. " +
        "Một số dự đoán có ký tự ở vị trí không thông thường — có thể do lỗi mô hình, ảnh LR quá mờ hoặc biển số thực không tuân theo pattern phổ biến.",
        { x: 5.3, y: 5.55, w: 7.3, h: 1.1, fontSize: 12.5, color: MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    pageNumber(slide, 18);
}

// =====================================================================
// SLIDE 19 - Kết quả inference - ví dụ
// =====================================================================
{
    const slide = lightSlide("Kết quả đánh giá", "Kết quả Inference — Ví dụ minh họa");

    slide.addText(
        "Bảng kết quả suy luận chứa 2,001 dòng dự đoán. Một số mẫu quan sát được:",
        { x: 0.7, y: 1.6, w: 12.0, h: 0.4, fontSize: 15, color: TEXT_DARK, fontFace: "Calibri" }
    );

    const rows = [
        [{ text: "Track ID", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } }, { text: "Dự đoán", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } }, { text: "Quan sát", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } }],
        ["track_00001_lr", "AVL5215", "Brazilian format, 3 chữ và 4 số"],
        ["track_02239_lr", "SFG1D23", "Mercosur format, có chữ ở vị trí thứ 5"],
        ["track_02251_lr", "BBX5C92", "Mercosur format"],
        ["track_00485_lr", "LVO8854", "Brazilian-like format"],
    ].map((row, i) => i === 0 ? row : row.map((c, ci) => ({ text: c, options: { color: ci === 1 ? NAVY : TEXT_DARK, bold: ci === 1, fontFace: ci === 1 ? "Courier New" : "Calibri", fill: { color: i % 2 === 0 ? "FFFFFF" : "F4F6FA" }, align: ci === 1 ? "center" : "left" } })));

    slide.addTable(rows, {
        x: 0.7, y: 2.2, w: 8.2, h: 2.2,
        colW: [2.7, 2.1, 3.4],
        fontSize: 14, fontFace: "Calibri", border: { pt: 0.5, color: "DCE4F0" },
        rowH: 0.5, valign: "middle",
    });

    // right summary card
    const rx = 9.25, rw = 3.4;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 2.2, w: rw, h: 4.55, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    iconCircle(slide, "carside", rx + 0.3, 2.45, 0.6, AMBER, "navy");
    slide.addText("2,001", { x: rx, y: 3.25, w: rw, h: 0.9, fontSize: 44, bold: true, color: "FFFFFF", align: "center", fontFace: "Cambria" });
    slide.addText("dòng dự đoán trong bảng kết quả", { x: rx + 0.3, y: 4.15, w: rw - 0.6, h: 0.6, fontSize: 13, color: "CADCFC", align: "center", fontFace: "Calibri", lineSpacingMultiple: 1.2 });
    slide.addShape(pres.shapes.LINE, { x: rx + 0.4, y: 4.95, w: rw - 0.8, h: 0, line: { color: "3E4E78", width: 1 } });
    slide.addText(
        "Phân tích blind test tập trung vào: thống kê output, tính hợp lệ của pattern và khả năng tạo chuỗi biển số theo hai định dạng chính.",
        { x: rx + 0.3, y: 5.15, w: rw - 0.6, h: 1.45, fontSize: 12.5, color: "E6ECF7", align: "left", fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    // bottom note
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 4.65, w: 8.2, h: 2.1, rectRadius: 0.08, fill: { color: "FFF6DE" }, shadow: shadow() });
    slide.addImage({ path: iconPath("warn", "navy"), x: 1.0, y: 4.9, w: 0.5, h: 0.5 });
    slide.addText(
        "Một số dự đoán có ký tự ở vị trí không thông thường, có thể đến từ lỗi mô hình, ảnh LR quá mờ hoặc trường hợp biển số thực " +
        "không tuân theo pattern phổ biến. Đây là cơ sở cho các phân tích định tính và ablation tiếp theo.",
        { x: 1.65, y: 4.9, w: 7.05, h: 1.7, fontSize: 13.5, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.3, valign: "middle" }
    );

    pageNumber(slide, 19);
}

// =====================================================================
// SLIDE 20 - Phân tích định tính: điểm mạnh & hạn chế
// =====================================================================
{
    const slide = lightSlide("Phân tích định tính", "Điểm mạnh quan sát được & Hạn chế còn tồn tại");

    // strengths
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.65, w: 6.0, h: 5.1, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
    iconCircle(slide, "check", 1.0, 1.9, 0.6, "DCF5E3", "navy");
    slide.addText("Điểm mạnh", { x: 1.75, y: 1.9, w: 4.5, h: 0.6, fontSize: 17, bold: true, color: "1F7A3D", fontFace: "Cambria", valign: "middle" });

    const strengths = [
        "Khai thác nhiều frame thay vì phụ thuộc một ảnh duy nhất, phù hợp dữ liệu dạng video track.",
        "STN giúp hệ thống học căn chỉnh hình học, giảm ảnh hưởng lệch góc và biến dạng phối cảnh.",
        "ResNet pretrained cung cấp đặc trưng thị giác mạnh trong điều kiện ảnh độ phân giải thấp.",
        "Transformer Encoder mô hình hóa quan hệ giữa các vị trí ký tự, giảm lỗi nhận dạng cục bộ.",
        "Kiến trúc modular cho phép thay thế backbone hoặc sequence model để thử nghiệm nhanh.",
    ];
    let sy = 2.7;
    strengths.forEach(t => {
        slide.addImage({ path: iconPath("check", "navy"), x: 1.0, y: sy + 0.05, w: 0.3, h: 0.3 });
        slide.addText(t, { x: 1.45, y: sy - 0.05, w: 5.0, h: 0.75, fontSize: 13, color: TEXT_DARK, fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.2 });
        sy += 0.85;
    });

    // limitations
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.9, y: 1.65, w: 5.75, h: 5.1, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    iconCircle(slide, "warn", 7.2, 1.9, 0.6, AMBER, "navy");
    slide.addText("Hạn chế còn tồn tại", { x: 7.95, y: 1.9, w: 4.5, h: 0.6, fontSize: 17, bold: true, color: AMBER, fontFace: "Cambria", valign: "middle" });

    const limits = [
        "Chất lượng ảnh LR quá thấp khiến ký tự hình dạng tương tự dễ nhầm: O/0, I/1, B/8, S/5.",
        "Mất cân bằng dữ liệu trong Scenario-B có thể khiến mô hình ưu tiên pattern Mercosur hơn Brazilian.",
        "Phân loại độc lập 7 vị trí ký tự giả định độ dài biển số cố định, chưa linh hoạt như CTC hoặc attention decoder cho chuỗi biến độ dài.",
        "Chưa có điểm leaderboard chính thức trên blind test, chưa thể kết luận tuyệt đối về hiệu năng cuối cùng.",
    ];
    let ly = 2.7;
    limits.forEach(t => {
        slide.addShape(pres.shapes.OVAL, { x: 7.2, y: ly + 0.08, w: 0.12, h: 0.12, fill: { color: AMBER } });
        slide.addText(t, { x: 7.45, y: ly - 0.1, w: 4.95, h: 0.95, fontSize: 13, color: "E6ECF7", fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.2 });
        ly += 1.05;
    });

    pageNumber(slide, 20);
}

// =====================================================================
// SLIDE 21 - Ablation Study
// =====================================================================
{
    const slide = lightSlide("Ablation & Thử nghiệm mở rộng", "So sánh kiến trúc & Vai trò Attention Fusion");

    // left: architecture comparison
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.65, w: 6.0, h: 5.1, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
    slide.addText("So sánh các hướng kiến trúc", { x: 1.0, y: 1.9, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: TEXT_DARK, fontFace: "Cambria" });

    const archs = [
        ["ResNet-50 + Transformer", "Cấu hình chính — backbone sâu cung cấp đặc trưng thị giác mạnh, Transformer phù hợp với chuỗi đặc trưng theo chiều ngang.", AMBER],
        ["ResNet-34 + Transformer", "Backbone nhẹ hơn — kiểm tra đánh đổi giữa số tham số và khả năng nhận dạng.", "EAF1FB"],
        ["ResNet-34 + BiLSTM", "Decoder thay thế — kiểm tra liệu mô hình recurrent nhẹ hơn có thể thay Transformer cho chuỗi ký tự ngắn.", "EAF1FB"],
    ];
    let ay = 2.5;
    archs.forEach(([title, desc, bg]) => {
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: ay, w: 5.4, h: 1.35, rectRadius: 0.08, fill: { color: bg } });
        slide.addText(title, { x: 1.2, y: ay + 0.12, w: 5.0, h: 0.4, fontSize: 14, bold: true, color: bg === AMBER ? NAVY : TEXT_DARK, fontFace: "Cambria" });
        slide.addText(desc, { x: 1.2, y: ay + 0.52, w: 5.0, h: 0.75, fontSize: 11.5, color: bg === AMBER ? NAVY : MUTED, fontFace: "Calibri", lineSpacingMultiple: 1.15 });
        ay += 1.5;
    });

    // right: attention fusion role
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.9, y: 1.65, w: 5.75, h: 5.1, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    slide.addText("Vai trò của Attention Fusion", { x: 7.2, y: 1.9, w: 5.2, h: 0.4, fontSize: 16, bold: true, color: AMBER, fontFace: "Cambria" });

    slide.addText(
        "Nếu không có fusion có trọng số, mô hình phải dựa vào một frame đơn lẻ hoặc phép gộp thô, trong khi các frame trong cùng track " +
        "thường có độ mờ, nhiễu và biến dạng khác nhau. Attention Fusion giúp mô hình học trọng số theo frame và theo vị trí đặc trưng.",
        { x: 7.2, y: 2.4, w: 5.2, h: 1.4, fontSize: 13, color: "E6ECF7", fontFace: "Calibri", lineSpacingMultiple: 1.3 }
    );

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y: 3.95, w: 5.2, h: 1.3, rectRadius: 0.08, fill: { color: SLATE } });
    slide.addText("Vị trí fusion: sau ResNet, trước Transformer Encoder.", { x: 7.45, y: 4.1, w: 4.7, h: 0.4, fontSize: 13, bold: true, color: AMBER, fontFace: "Calibri" });
    slide.addText(
        "Backbone đã chuyển ảnh thành đặc trưng giàu ngữ nghĩa; Transformer chỉ cần xử lý một chuỗi đại diện cho toàn bộ track — chi phí sequence modeling được giữ ổn định.",
        { x: 7.45, y: 4.5, w: 4.7, h: 0.7, fontSize: 12, color: "CADCFC", fontFace: "Calibri", lineSpacingMultiple: 1.2 }
    );

    slide.addText(
        "Nhận xét: Ablation không chỉ tìm accuracy cao nhất, mà giúp hiểu rõ vai trò từng thành phần — STN cho căn chỉnh, backbone cho đặc trưng ảnh, " +
        "fusion cho multi-frame, Transformer/BiLSTM cho ngữ cảnh chuỗi.",
        { x: 7.2, y: 5.45, w: 5.2, h: 1.2, fontSize: 12.5, italic: true, color: "E6ECF7", fontFace: "Calibri", lineSpacingMultiple: 1.25 }
    );

    pageNumber(slide, 21);
}

// =====================================================================
// SLIDE 22 - Thảo luận: kiểm chứng giả thuyết & rủi ro triển khai
// =====================================================================
{
    const slide = lightSlide("Thảo luận", "Kiểm chứng giả thuyết & Rủi ro triển khai thực tế");

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.65, w: 6.0, h: 5.1, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
    iconCircle(slide, "balance", 1.0, 1.9, 0.6, "EAF1FB", "slate");
    slide.addText("Kiểm chứng giả thuyết nghiên cứu", { x: 1.75, y: 1.9, w: 4.8, h: 0.6, fontSize: 15, bold: true, color: TEXT_DARK, fontFace: "Cambria", valign: "middle" });

    slide.addText(
        "Các phân tích cho thấy giả thuyết khai thác multi-frame phù hợp với bài toán LRLPR. Khi ảnh đơn lẻ quá mờ, hợp nhất đặc trưng từ " +
        "nhiều frame giúp mô hình có thêm thông tin bổ sung. STN và backbone pretrained giúp giảm chi phí học từ đầu và tăng khả năng tổng quát hóa.",
        { x: 1.0, y: 2.65, w: 5.4, h: 1.5, fontSize: 13.5, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.3 }
    );

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: 4.3, w: 5.4, h: 2.2, rectRadius: 0.08, fill: { color: "EAF1FB" } });
    slide.addText(
        "Tuy nhiên, bài toán vẫn chịu giới hạn bởi chất lượng ảnh LR và sự mất cân bằng dữ liệu. Cần thêm ablation định lượng với validation " +
        "exact-match accuracy cho từng biến thể kiến trúc và từng scenario để đánh giá đầy đủ hơn.",
        { x: 1.25, y: 4.3, w: 4.9, h: 2.2, fontSize: 13, italic: true, color: TEXT_DARK, fontFace: "Calibri", lineSpacingMultiple: 1.3, valign: "middle" }
    );

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.9, y: 1.65, w: 5.75, h: 5.1, rectRadius: 0.08, fill: { color: NAVY }, shadow: shadow() });
    iconCircle(slide, "warn", 7.2, 1.9, 0.6, AMBER, "navy");
    slide.addText("Rủi ro khi triển khai thực tế", { x: 7.95, y: 1.9, w: 4.6, h: 0.6, fontSize: 15, bold: true, color: AMBER, fontFace: "Cambria", valign: "middle" });

    const risks = [
        "Lỗi nhận dạng một ký tự có thể dẫn đến sai toàn bộ biển số.",
        "Khó khăn với biển số bẩn, che khuất, phản sáng hoặc không đúng định dạng phổ biến.",
        "Hệ thống nên kết hợp thêm kiểm tra pattern, confidence score và hậu xử lý theo quy tắc biển số từng quốc gia.",
    ];
    let ry = 2.7;
    risks.forEach(t => {
        slide.addShape(pres.shapes.OVAL, { x: 7.2, y: ry + 0.08, w: 0.12, h: 0.12, fill: { color: AMBER } });
        slide.addText(t, { x: 7.45, y: ry - 0.1, w: 4.95, h: 1.0, fontSize: 13.5, color: "E6ECF7", fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.25 });
        ry += 1.15;
    });

    pageNumber(slide, 22);
}

// =====================================================================
// SLIDE 23 - Kết luận
// =====================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: NAVY };

    slide.addText("KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", { x: 0.7, y: 0.4, w: 10, h: 0.35, fontSize: 13, color: AMBER, bold: true, fontFace: "Calibri", charSpacing: 2 });
    slide.addText("Kết luận", { x: 0.7, y: 0.7, w: 12, h: 0.8, fontSize: 30, color: "FFFFFF", bold: true, fontFace: "Cambria" });

    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.7, w: 6.1, h: 5.05, rectRadius: 0.08, fill: { color: SLATE } });
    slide.addText(
        "Đề tài tập trung vào nhận dạng biển số xe độ phân giải thấp trong bối cảnh dữ liệu video track. Hệ thống " +
        "LowResolution-LPR đề xuất kiến trúc ResTranOCR, kết hợp STN, ResNet, Attention Fusion và Transformer Encoder để " +
        "nhận dạng chuỗi biển số từ nhiều frame LR — phù hợp đặc trưng của cuộc thi ICPR 2026 LRLPR.",
        { x: 1.0, y: 1.95, w: 5.5, h: 2.0, fontSize: 14, color: "FFFFFF", fontFace: "Calibri", lineSpacingMultiple: 1.35 }
    );
    slide.addText(
        "Báo cáo đã trình bày bối cảnh, dữ liệu, pipeline huấn luyện, kiến trúc mô hình, kết quả suy luận và các hướng " +
        "thử nghiệm mở rộng. Mặc dù chưa có điểm leaderboard blind test chính thức, cấu trúc hệ thống cho thấy một hướng " +
        "tiếp cận hợp lý và có khả năng mở rộng.",
        { x: 1.0, y: 4.1, w: 5.5, h: 2.4, fontSize: 14, color: "CADCFC", fontFace: "Calibri", lineSpacingMultiple: 1.35, italic: true }
    );

    // future directions
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 1.7, w: 5.65, h: 5.05, rectRadius: 0.08, fill: { color: CARD } });
    slide.addText("Hướng phát triển", { x: 7.3, y: 1.9, w: 5, h: 0.4, fontSize: 17, bold: true, color: TEXT_DARK, fontFace: "Cambria" });

    const futures = [
        ["search", "Đánh giá định lượng chi tiết từng thành phần ResTranOCR (STN, Attention Fusion, Transformer Encoder)."],
        ["random", "Ensemble nhiều cấu hình ResNet và decoder khác nhau để tăng độ ổn định dự đoán."],
        ["arrows", "Áp dụng CTC Loss hoặc attention decoder để xử lý chuỗi linh hoạt thay vì cố định 7 ký tự."],
        ["idcard", "Hoàn thiện hậu xử lý theo pattern Brazilian/Mercosur, đặc biệt ở ký tự thứ 5."],
        ["bar", "Đánh giá chi tiết theo scenario, loại biển và độ phân giải để xác định nhóm dữ liệu còn yếu."],
    ];
    let fy2 = 2.5;
    futures.forEach(([icon, t]) => {
        iconCircle(slide, icon, 7.3, fy2, 0.5, NAVY, "amber");
        slide.addText(t, { x: 7.95, y: fy2 - 0.07, w: 4.4, h: 0.75, fontSize: 12.5, color: TEXT_DARK, fontFace: "Calibri", valign: "middle", lineSpacingMultiple: 1.2 });
        fy2 += 0.92;
    });

    pageNumber(slide, 23);
}

// =====================================================================
// SLIDE 24 - Tài liệu tham khảo
// =====================================================================
{
    const slide = lightSlide("Tài liệu tham khảo", "References");

    iconCircle(slide, "book", 0.7, 2.0, 0.0, LIGHT, "navy"); // placeholder no-op (size 0 avoided below)
    const refs = [
        "ICPR 2026 Low-Resolution License Plate Recognition Challenge — trang mô tả cuộc thi.",
        "M. Jaderberg et al., \"Spatial Transformer Networks\", NeurIPS, 2015.",
        "K. He et al., \"Deep Residual Learning for Image Recognition\", CVPR, 2016.",
        "A. Vaswani et al., \"Attention Is All You Need\", NeurIPS, 2017.",
        "A. Graves et al., \"Connectionist Temporal Classification\", ICML, 2006.",
    ];
    let ry2 = 1.9;
    refs.forEach((r, i) => {
        slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: ry2, w: 12.0, h: 0.85, rectRadius: 0.08, fill: { color: CARD }, shadow: shadow() });
        slide.addShape(pres.shapes.OVAL, { x: 1.0, y: ry2 + 0.2, w: 0.45, h: 0.45, fill: { color: NAVY } });
        slide.addText(String(i + 1), { x: 1.0, y: ry2 + 0.2, w: 0.45, h: 0.45, fontSize: 14, bold: true, color: AMBER, align: "center", valign: "middle", fontFace: "Cambria" });
        slide.addText(r, { x: 1.65, y: ry2, w: 10.8, h: 0.85, fontSize: 14, color: TEXT_DARK, fontFace: "Calibri", valign: "middle" });
        ry2 += 1.0;
    });

    pageNumber(slide, 24);
}

// =====================================================================
pres.writeFile({ fileName: "/home/claude/lpr_deck/output.pptx" }).then(() => console.log("done"));