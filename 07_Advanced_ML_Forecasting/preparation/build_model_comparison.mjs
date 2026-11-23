import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = new URL(".", import.meta.url).pathname;
const outDir = `${root}outputs/`;
const xlsxPath = `${root}ServeSmart_Phase2_Model_Comparison.xlsx`;

function parseCSV(text) {
  const rows=[]; let row=[]; let cell=""; let quoted=false;
  for (let i=0;i<text.length;i++) {
    const ch=text[i], next=text[i+1];
    if (ch==='"') { if (quoted && next==='"') { cell+='"'; i++; } else quoted=!quoted; }
    else if (ch===',' && !quoted) { row.push(cell); cell=""; }
    else if ((ch==='\n' || ch==='\r') && !quoted) { if (ch==='\r' && next==='\n') i++; row.push(cell); rows.push(row); row=[]; cell=""; }
    else cell+=ch;
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  const headers=rows.shift();
  return rows.filter(r=>r.length===headers.length).map(r=>Object.fromEntries(headers.map((h,i)=>[h,r[i]])));
}
function matrix(records) {
  if (!records.length) return [["No rows"]];
  const headers=Object.keys(records[0]);
  return [headers, ...records.map(r=>headers.map(h=>{
    const v=r[h]; if (v==="") return null; if (v!==undefined && /^-?\d+(\.\d+)?$/.test(v)) return Number(v); return v ?? null;
  }))];
}
function styleSheet(sheet, range, headerRange) {
  sheet.getRange(range).format.font = { name:"Arial", size:10 };
  sheet.getRange(headerRange).format = { fill:"#1F4E78", font:{name:"Arial", bold:true, color:"#FFFFFF"} };
  sheet.freezePanes.freezeRows(1); sheet.showGridLines=false;
}

const metrics = parseCSV(await fs.readFile(`${outDir}model_metrics.csv`,"utf8"));
const checks = parseCSV(await fs.readFile(`${outDir}data_quality_checks.csv`,"utf8"));
const deps = parseCSV(await fs.readFile(`${outDir}dependency_status.csv`,"utf8"));
const predictors = parseCSV(await fs.readFile(`${outDir}predictor_manifest.csv`,"utf8"));
const availability = parseCSV(await fs.readFile(`${outDir}predictor_availability.csv`,"utf8"));
const summary = JSON.parse(await fs.readFile(`${outDir}run_summary.json`,"utf8"));

const wb=Workbook.create();
const s=wb.worksheets.add("Summary");
s.getRange("A1:F1").merge(); s.getRange("A1").values=[["ServeSmart AI — Phase 2 Model Comparison"]];
s.getRange("A1").format={fill:"#17365D",font:{name:"Arial",size:16,bold:true,color:"#FFFFFF"}};
s.getRange("A3:B10").values=[
 ["Input / execution check","Result"],["Source hourly rows",summary.source_rows],["Target units",summary.source_units],["Hourly fit rows",summary.hourly_fit_rows],["Negative target rows excluded from hourly fit",summary.hourly_negative_excluded],["Observed-zero hours",summary.observed_zero],["Spring Break predictor", "Excluded; historical dates retained"],["Advanced ML status","Blocked by missing optional dependencies"]
];
s.getRange("A3:B3").format={fill:"#5B9BD5",font:{name:"Arial",bold:true,color:"#FFFFFF"}};
s.getRange("A12:F12").values=[["Forecast level","Model","Evaluation","Observations","MAE","RMSE"]];
const overall=metrics.filter(r=>r.Evaluation==="Common eligible test rows");
s.getRange(`A13:F${12+overall.length}`).values=overall.map(r=>[r.ForecastLevel,r.Model,r.Evaluation,Number(r.Observations),Number(r.MAE),Number(r.RMSE)]);
s.getRange(`G12:G${12+overall.length}`).values=[["WAPE"],...overall.map(r=>[Number(r.WAPE)])];
s.getRange("A12:G12").format={fill:"#5B9BD5",font:{name:"Arial",bold:true,color:"#FFFFFF"}};
s.getRange(`E13:F${12+overall.length}`).format.numberFormat="0.00"; s.getRange(`G13:G${12+overall.length}`).format.numberFormat="0.0%";
s.getRange("A1:G30").format.font={name:"Arial",size:10}; s.getRange("A1").format={fill:"#17365D",font:{name:"Arial",size:16,bold:true,color:"#FFFFFF"}}; s.showGridLines=false;

const m=wb.worksheets.add("Metrics"); const mm=matrix(metrics); m.getRangeByIndexes(0,0,mm.length,mm[0].length).values=mm; styleSheet(m,`A1:${String.fromCharCode(64+mm[0].length)}${mm.length}`,`A1:${String.fromCharCode(64+mm[0].length)}1`);
const c=wb.worksheets.add("Checks"); const cm=matrix(checks); c.getRangeByIndexes(0,0,cm.length,cm[0].length).values=cm; styleSheet(c,`A1:D${cm.length}`,"A1:D1");
const d=wb.worksheets.add("Dependencies"); const dm=matrix(deps); d.getRangeByIndexes(0,0,dm.length,dm[0].length).values=dm; styleSheet(d,`A1:C${dm.length}`,"A1:C1");
const p=wb.worksheets.add("Predictors"); const pm=matrix(predictors); p.getRangeByIndexes(0,0,pm.length,pm[0].length).values=pm; styleSheet(p,`A1:C${pm.length}`,"A1:C1");
const a=wb.worksheets.add("Availability"); const am=matrix(availability); a.getRangeByIndexes(0,0,am.length,am[0].length).values=am; styleSheet(a,`A1:D${am.length}`,"A1:D1");

for (const sh of [s,m,c,d,p]) { const used=sh.getUsedRange(); used.format.autofitColumns(); }
wb.recalculate();
const preview=await wb.render({sheetName:"Summary",autoCrop:"all",scale:1,format:"png"});
await fs.writeFile(`${root}summary_preview.png`,new Uint8Array(await preview.arrayBuffer()));
const xlsx=await SpreadsheetFile.exportXlsx(wb); await xlsx.save(xlsxPath);
console.log(JSON.stringify({xlsxPath, sheets:["Summary","Metrics","Checks","Dependencies","Predictors","Availability"]}));
