"""
Generate the three knowledge-base PDFs for FunStay's RAG.

1. ChampionsGate Property Manual: includes a Pool Heating Cost table (complex tabular data
   that the OCR should extract well)
2. Reunion Welcome Book: includes simulated "handwritten" note from Mike (italic + cursive-ish font)
3. Orlando Area Guide: distance table to parks + restaurant + grocery recommendations
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUT_DIR = "/home/claude/funstay_webhook/docs"
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=22, spaceAfter=20)
h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=15, spaceAfter=10, textColor=colors.HexColor('#1F3A93'))
h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceAfter=6)
body = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=10.5, leading=14, spaceAfter=8)
handwritten = ParagraphStyle('Hand', parent=styles['BodyText'], fontName='Helvetica-Oblique',
                              fontSize=11.5, leading=16, textColor=colors.HexColor('#2C3E50'),
                              leftIndent=20, rightIndent=20, spaceAfter=10, spaceBefore=10,
                              borderPadding=10, backColor=colors.HexColor('#FFF8DC'),
                              borderColor=colors.HexColor('#D4A017'), borderWidth=1)


# ============================================================
# PDF 1 — ChampionsGate Property Manual
# ============================================================
doc1_path = f"{OUT_DIR}/01_ChampionsGate_Property_Manual.pdf"
doc1 = SimpleDocTemplate(doc1_path, pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch)
story = []

story.append(Paragraph("FunStay Florida", title_style))
story.append(Paragraph("ChampionsGate Villa — Property Manual", h1))
story.append(Paragraph("Address: 8924 Bismarck Palm Dr, Davenport, FL 33896 — Resort: ChampionsGate Country Club", body))
story.append(Spacer(1, 12))

# ---- Pool & Spa section ----
story.append(Paragraph("1. Pool and Hot Tub (Spa) Heating", h1))
story.append(Paragraph(
    "The pool and spa are heated by a natural-gas heater controlled from the lanai panel. "
    "Heating is <b>NOT included</b> in the nightly rate and is billed as an optional add-on. "
    "Activation must be requested at least 24 hours in advance via the FunStay concierge so the heater can be primed.",
    body))

story.append(Paragraph("Heating cost and warm-up times by season", h2))
pool_table_data = [
    ["Season", "Months", "Daily cost (USD)", "Warm-up time", "Target temp"],
    ["Winter", "Dec–Feb", "$35 / day", "12–18 h", "86°F (30°C)"],
    ["Spring", "Mar–May", "$28 / day", "8–12 h", "86°F (30°C)"],
    ["Summer", "Jun–Aug", "$18 / day", "4–6 h", "88°F (31°C)"],
    ["Fall", "Sep–Nov", "$25 / day", "8–10 h", "86°F (30°C)"],
]
t = Table(pool_table_data, colWidths=[0.9*inch, 1.0*inch, 1.3*inch, 1.2*inch, 1.3*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A93')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9.5),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F5F5F5'), colors.white]),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('PADDING', (0, 0), (-1, -1), 5),
]))
story.append(t)
story.append(Spacer(1, 12))

story.append(Paragraph(
    "<b>Spa (jacuzzi) operation</b>: The jacuzzi shares the same heater system. Once the pool is heated, "
    "the jacuzzi reaches operating temperature (102°F / 39°C) in approximately 1 hour. "
    "Use the white button on the lanai panel to activate jets — they run for 30 minutes automatically. "
    "Spa hours are restricted to <b>8:00 AM – 11:00 PM</b> per resort regulations.",
    body))
story.append(Paragraph(
    "<b>Safety</b>: Children under 14 must be supervised at all times. Glass containers are prohibited in "
    "the pool and spa area. Do not enter the pool during thunderstorms.",
    body))

story.append(Spacer(1, 16))

# ---- WiFi & Smart Home ----
story.append(Paragraph("2. WiFi and Smart Home Devices", h1))
story.append(Paragraph(
    "WiFi network: <b>FunStay_Guest</b> · Password: <b>Orlando2026!</b><br/>"
    "The router is located in the master bedroom closet. If you lose connection, unplug the router for 30 "
    "seconds and plug it back in.",
    body))
story.append(Paragraph(
    "Smart TVs (Samsung) in living room and all bedrooms are pre-logged into Netflix, Disney+, and Hulu under "
    "FunStay's account. Please do not change account settings.",
    body))

story.append(Spacer(1, 12))

# ---- Climate Control ----
story.append(Paragraph("3. Climate Control (AC)", h1))
story.append(Paragraph(
    "The Nest thermostat in the hallway controls all zones. Recommended setting: <b>72°F (22°C)</b> in summer, "
    "<b>68°F (20°C)</b> in winter. Setting the AC below 68°F may freeze the unit and trigger a service charge.",
    body))

# ---- Trash ----
story.append(Paragraph("4. Trash and Recycling", h1))
story.append(Paragraph(
    "Trash pickup at ChampionsGate is on <b>Tuesday and Friday mornings</b>. Place the gray bin at the curb "
    "by 7:00 AM. Recycling (blue bin) is collected on <b>Wednesdays</b>.",
    body))

doc1.build(story)
print(f"✓ Created: {doc1_path}")


# ============================================================
# PDF 2 — Reunion Welcome Book (with simulated "handwritten" note)
# ============================================================
doc2_path = f"{OUT_DIR}/02_Reunion_Welcome_Book.pdf"
doc2 = SimpleDocTemplate(doc2_path, pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch)
story = []

story.append(Paragraph("FunStay Florida", title_style))
story.append(Paragraph("Reunion Resort — Welcome Book", h1))
story.append(Paragraph("Address: 7593 Gathering Dr, Reunion, FL 34747 — Resort: Reunion Resort & Golf Club", body))
story.append(Spacer(1, 12))

# Handwritten note from Mike
story.append(Paragraph("A note from Mike, your host", h2))
story.append(Paragraph(
    "Hey there, and welcome to Reunion! I&#39;m Mike — I&#39;ve been hosting families here since 2019. "
    "A few things I always tell my guests in person: the gate code is <b>4477</b> (push the # key after). "
    "If the gate doesn&#39;t open, it&#39;s usually because you waited too long after entering the code — "
    "just try again. The closest Publix is 8 minutes away on Sinclair Rd, and they have an amazing deli counter. "
    "If you need ANYTHING during your stay, call or text me at <b>(321) 430-8681</b>. I&#39;m local and I "
    "respond fast. Have an incredible time. — Mike",
    handwritten))
story.append(Spacer(1, 16))

# ---- Gate codes ----
story.append(Paragraph("1. Resort Gate Codes", h1))
gate_data = [
    ["Gate", "Code", "Hours staffed"],
    ["Main entrance (Reunion Blvd)", "4477", "24/7"],
    ["North gate (Patrician Way)", "8821", "6 AM – 10 PM"],
    ["Pool & clubhouse parking", "1212", "6 AM – 11 PM"],
]
t = Table(gate_data, colWidths=[2.6*inch, 1.0*inch, 1.6*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A93')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F5F5F5'), colors.white]),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
]))
story.append(t)
story.append(Spacer(1, 14))

# ---- HOA rules ----
story.append(Paragraph("2. Reunion Resort HOA Rules", h1))
story.append(Paragraph(
    "Reunion has stricter HOA enforcement than other Disney-area resorts. Please respect these rules to avoid fines.",
    body))
story.append(Paragraph(
    "• <b>Quiet hours</b>: 10:00 PM to 8:00 AM. Outdoor music must be off by 9:00 PM.<br/>"
    "• <b>Parking</b>: Maximum 4 vehicles per villa. No street parking overnight (fines start at $100).<br/>"
    "• <b>Visitors</b>: Day visitors must be registered at the main gate. No commercial vehicles after 6 PM.<br/>"
    "• <b>Pool deck</b>: No glass containers. Towels must remain at your villa, not the community pool.",
    body))

story.append(Spacer(1, 12))

# ---- Amenities ----
story.append(Paragraph("3. Resort Amenities", h1))
story.append(Paragraph(
    "Reunion has three signature golf courses (Palmer, Watson, Nicklaus), a 5-acre water park (<b>Cove Pool & Lazy River</b>), "
    "and a full-service spa. Guest passes for the water park are <b>$10/person/day</b> and can be purchased at the clubhouse. "
    "The lazy river is open from 10 AM to 8 PM, daily.",
    body))

doc2.build(story)
print(f"✓ Created: {doc2_path}")


# ============================================================
# PDF 3 — Orlando Area Guide
# ============================================================
doc3_path = f"{OUT_DIR}/03_Orlando_Area_Guide.pdf"
doc3 = SimpleDocTemplate(doc3_path, pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch)
story = []

story.append(Paragraph("FunStay Florida", title_style))
story.append(Paragraph("Orlando Area Guide for FunStay Guests", h1))
story.append(Spacer(1, 10))

# ---- Distance table ----
story.append(Paragraph("1. Distances and Drive Times from FunStay Properties", h1))
story.append(Paragraph(
    "All times assume no major traffic. Add 15–25 minutes during park opening and closing hours (8–10 AM, 8–10 PM).",
    body))

dist_data = [
    ["Destination", "From ChampionsGate", "From Reunion", "From Windsor Hills", "From Solterra"],
    ["Walt Disney World — Magic Kingdom", "12 mi · 20 min", "8 mi · 15 min", "5 mi · 12 min", "14 mi · 22 min"],
    ["Disney Springs", "10 mi · 18 min", "9 mi · 17 min", "6 mi · 13 min", "12 mi · 20 min"],
    ["Universal Studios", "20 mi · 30 min", "21 mi · 32 min", "18 mi · 27 min", "22 mi · 33 min"],
    ["SeaWorld", "18 mi · 28 min", "19 mi · 29 min", "16 mi · 24 min", "20 mi · 30 min"],
    ["Orlando Airport (MCO)", "25 mi · 35 min", "26 mi · 36 min", "23 mi · 33 min", "28 mi · 38 min"],
    ["Premium Outlets — Vineland", "15 mi · 22 min", "13 mi · 20 min", "11 mi · 17 min", "17 mi · 25 min"],
]
t = Table(dist_data, colWidths=[2.0*inch, 1.2*inch, 1.1*inch, 1.2*inch, 1.1*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A93')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8.5),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F5F5F5'), colors.white]),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
story.append(t)
story.append(Spacer(1, 14))

# ---- Transport options ----
story.append(Paragraph("2. Transportation Options", h1))
story.append(Paragraph(
    "<b>Rental car</b>: Strongly recommended for stays of 3+ days. Disney has free parking for resort guests "
    "and on-property hotels, but FunStay villa guests pay $30/day for theme park parking.<br/><br/>"
    "<b>Disney transportation</b>: Free Disney buses, monorail, and Skyliner are available <i>only</i> to "
    "guests staying on Disney property. FunStay guests must drive or rideshare to the parks.<br/><br/>"
    "<b>Uber/Lyft to MCO</b>: Approximately $45–65 from ChampionsGate, $50–70 from Reunion. Schedule "
    "30 minutes earlier than you think you need during peak hours.<br/><br/>"
    "<b>Mears Connect shuttle</b>: Scheduled airport shuttle service, $32/person round-trip. Book at "
    "mearsconnect.com 24h in advance.",
    body))

story.append(PageBreak())

# ---- Restaurants ----
story.append(Paragraph("3. Restaurant Recommendations", h1))

story.append(Paragraph("Family-friendly (kids welcome, no reservation needed)", h2))
story.append(Paragraph(
    "• <b>Giordano&#39;s Pizza</b> (8865 Commodity Cir) — Chicago deep-dish, hearty portions. ~$20/person.<br/>"
    "• <b>Twistee Treat</b> (9000 W Irlo Bronson Memorial Hwy) — Iconic ice cream cone-shaped shop. "
    "Cash and card. Kids love it.<br/>"
    "• <b>Miller&#39;s Ale House</b> (5505 W Irlo Bronson Memorial Hwy) — American sports bar, large menu, "
    "open until midnight.",
    body))

story.append(Paragraph("Date night / nicer dinners", h2))
story.append(Paragraph(
    "• <b>The Venetian Chop House</b> (Caribe Royale Orlando) — Italian-American steakhouse. Reservations required. ~$80/person.<br/>"
    "• <b>Christini&#39;s Ristorante Italiano</b> (7600 Dr Phillips Blvd) — Old-school Italian fine dining, "
    "white tablecloth. ~$100/person.<br/>"
    "• <b>Bahama Breeze</b> (8849 International Dr) — Caribbean-themed, oceanfront vibe, live music weekends.",
    body))

# ---- Groceries ----
story.append(Paragraph("4. Grocery Stores and Essentials", h1))
groc_data = [
    ["Store", "Address", "Hours", "Notes"],
    ["Publix Super Market", "1471 Champions Gate Blvd", "7 AM – 10 PM daily", "Closest to ChampionsGate · deli + bakery"],
    ["Walmart Supercenter", "1471 E Osceola Pkwy", "6 AM – 11 PM daily", "Cheapest groceries · 24/7 pharmacy"],
    ["Target", "3200 Rolling Oaks Blvd", "8 AM – 10 PM daily", "Good for last-minute items"],
    ["Aldi", "1450 W Vine St", "9 AM – 8 PM daily", "Discount groceries · bring quarter for cart"],
]
t = Table(groc_data, colWidths=[1.6*inch, 2.1*inch, 1.4*inch, 1.7*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A93')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F5F5F5'), colors.white]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('PADDING', (0, 0), (-1, -1), 5),
]))
story.append(t)

# ---- Emergency ----
story.append(Spacer(1, 14))
story.append(Paragraph("5. Emergency Contacts", h1))
story.append(Paragraph(
    "• <b>Emergency (police, fire, ambulance)</b>: 911<br/>"
    "• <b>FunStay 24/7 host (Mike)</b>: (321) 430-8681<br/>"
    "• <b>AdventHealth Celebration ER</b> (closest urgent care): 400 Celebration Pl, Celebration, FL · (407) 303-4000<br/>"
    "• <b>Centra Care urgent care</b> (24/7): 6918 W Irlo Bronson Memorial Hwy, Kissimmee · (407) 397-7032",
    body))

doc3.build(story)
print(f"✓ Created: {doc3_path}")

print("\nAll three PDFs created in", OUT_DIR)
