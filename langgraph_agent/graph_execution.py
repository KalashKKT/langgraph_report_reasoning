
from graph import system_graph
from graph import run_document_graph


pages = [

"Page 0: Reliance Industries Limited operates across energy, petrochemicals, retail, digital services (Jio), and new energy. Over the past decade, the company has diversified from a traditional refining business into consumer and technology-driven verticals.",

"Page 1: The Oil-to-Chemicals (O2C) segment contributes 45% of EBITDA. Refining margins have improved due to favorable global spreads and operational efficiencies.",

"Page 2: The Jio Platforms digital services segment contributes 28% of EBITDA and continues to add subscribers. ARPU increased 11% year-over-year, supported by tariff hikes and data consumption growth.",

"Page 3: Reliance Retail contributes 22% of EBITDA. Store expansion and higher same-store sales growth have driven a 19% revenue CAGR over the last three years.",

"Page 4: The New Energy division is investing in green hydrogen, solar manufacturing, and battery storage. Management expects this segment to become a meaningful EBITDA contributor over the next 5–7 years.",

"Page 5: Over the past 10 years, revenue has grown at a steady CAGR, with improving EBITDA margins driven by digital and retail expansion. Net profit growth has been supported by operating leverage and diversification.",

"Page 6: Return ratios such as ROE and ROCE have remained stable, reflecting disciplined capital allocation despite heavy capex in telecom and retail.",

"Page 7: The balance sheet shows significant gross debt due to expansion projects, but strong cash flows and periodic stake sales have supported net debt reduction and liquidity strength.",

"Page 8: Operating cash flow generation remains strong, although free cash flow fluctuates depending on annual capital expenditure intensity across segments.",

"Page 9: Current valuation metrics include P/E ratio, EV/EBITDA, Price-to-Book, and PEG ratio, which should be compared against historical averages and industry peers to determine relative valuation.",

"Page 10: Key growth drivers include Jio subscriber additions and ARPU expansion, Retail footprint growth and same-store sales momentum, O2C margin cycles, and long-term optionality from the New Energy investments.",

"Page 11: Key risks include refining margin volatility, telecom pricing competition, execution risks in New Energy projects, regulatory changes, and high capital expenditure requirements."

]


import json

# with open(r"D:\Piratech\data_agents\reports\text_extraction\aum_report.json", "r", encoding="utf-8") as f:
#     pages = json.load(f)


query = "Given that Reliance’s net profit growth in Q3FY26 was only 0.57% YoY despite strong performance in Jio and O2C segments, is the company entering a phase of margin compression that could limit earnings growth in the near term?"

result = run_document_graph(
    pages=pages,
    user_query=query,
    system_graph=system_graph
)

print(result["final_answer"])
print(result["validation_status"])