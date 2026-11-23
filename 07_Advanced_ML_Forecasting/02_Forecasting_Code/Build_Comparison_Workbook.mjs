import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const data=JSON.parse(await fs.readFile(path.join(root,'09_Documentation/Workbook_Content.json'),'utf8'));
const wb=Workbook.create();
const overview=wb.worksheets.add('Overview');
const sheets=[];
for (const item of data.sheets){
 const sh=wb.worksheets.add(item.name);sheets.push([sh,item]);
 sh.showGridLines=false;sh.tabColor=item.name==='Hourly'||item.name==='Daily'?'#187D91':'#667788';
 const count=item.columns.length,last=String.fromCharCode(64+count),n=item.rows.length;
 sh.getRange(`A1:${last}1`).merge();sh.getRange('A1').values=[[item.title]];sh.getRange(`A1:${last}1`).format={fill:'#15384A',font:{name:'Aptos',size:18,bold:true,color:'#FFFFFF'},rowHeight:34};
 sh.getRange(`A2:${last}3`).merge();sh.getRange('A2').values=[[item.subtitle]];sh.getRange(`A2:${last}3`).format={wrapText:true,font:{size:11,color:'#465D6B'},rowHeight:22};
 sh.getRangeByIndexes(4,0,1,count).values=[item.columns];sh.getRangeByIndexes(4,0,1,count).format={fill:'#DDECF0',font:{bold:true,color:'#15384A',size:11},wrapText:true,rowHeight:34};
 if(n)sh.getRangeByIndexes(5,0,n,count).values=item.rows;
 sh.getRangeByIndexes(5,0,Math.max(1,n),count).format.font={name:'Aptos',size:11};
 sh.getRange(`A1:${last}${n+5}`).format.columnWidth=19;
 sh.getRange(`A6:${last}${n+5}`).format.rowHeight=24;
 for(let j=0;j<count;j++){
  const col=item.columns[j];if(['MAE','RMSE','ValidationMAE','P90AbsoluteError'].includes(col))sh.getRangeByIndexes(5,j,n,1).setNumberFormat('0.00');
  if(col==='WAPE')sh.getRangeByIndexes(5,j,n,1).setNumberFormat('0.00"%"');
 }
 if(item.name==='Requirements'){
  sh.getRange('B:B').format.columnWidth=40;sh.getRange('C:C').format.columnWidth=32;sh.getRange('D:D').format.columnWidth=100;
  sh.getRange('A6:D14').format.wrapText=true;sh.getRange('A6:D14').format.rowHeight=76;
 } else if(item.name==='A B New') {sh.getRange('A:A').format.columnWidth=55;sh.getRange(`A6:A${n+5}`).format.wrapText=true;sh.getRange(`A6:G${n+5}`).format.rowHeight=38;}
 else {sh.getRange('A:A').format.columnWidth=25;sh.getRange('B:B').format.columnWidth=26;}
 if(item.name==='Daily approaches')sh.getRange('A:A').format.columnWidth=31;
 // Make long labels wrap instead of overflowing into adjacent numeric cells.
 for(let j=0;j<count;j++){
  const col=item.columns[j],rg=sh.getRangeByIndexes(5,j,n,1);
  if(['Model','FeatureSet','Location','Weekday','Source','Approach','Status','RequirementGroup','EvidenceAndLimit','ForecastLevel'].includes(col)){rg.format.wrapText=true;rg.format.horizontalAlignment='left';}
 }
 if(item.name!=='Requirements'&&item.name!=='A B New')sh.getRange(`A6:${last}${n+5}`).format.rowHeight=34;
 if(['By Location','By Weekday','By Hour'].includes(item.name)){
  sh.getRange('B:B').format.columnWidth=36;sh.getRange('C:C').format.columnWidth=30;
  sh.getRange(`A6:H${n+5}`).format.font={name:'Arial',size:11};
  sh.getRange(`D6:H${n+5}`).format.horizontalAlignment='right';
 }
 sh.freezePanes.freezeRows(5);
}
overview.showGridLines=false;overview.tabColor='#15384A';overview.getRange('A1:H1').merge();overview.getRange('A1').values=[['ServeSmart AI — Official Forecasting Comparison']];overview.getRange('A1:H1').format={fill:'#15384A',font:{name:'Aptos',size:20,bold:true,color:'#FFFFFF'},rowHeight:38};
overview.getRange('A2:H3').merge();overview.getRange('A2').values=[['Research results verified; source/provenance limitations remain. Read the Requirements tab and final report before claiming all requirements are complete.']];overview.getRange('A2:H3').format={wrapText:true,fill:'#FFF1D2',font:{size:12,color:'#6F4818'},rowHeight:25};
overview.getRange('A5:H5').values=[['Level','Selected on validation','May N','MAE','RMSE','WAPE','Naive MAE','MAE improvement']];overview.getRange('A5:H5').format={fill:'#DDECF0',font:{bold:true},rowHeight:35,wrapText:true};
for(let i=0;i<data.selected.length;i++){
 const s=data.selected[i],d=data.improvement.find(x=>x.ForecastLevel===s.ForecastLevel&&x.Model===s.Model),row=i+6;
 overview.getRange(`A${row}:G${row}`).values=[[s.ForecastLevel,s.Model,s.Observations,s.MAE,s.RMSE,s.WAPE,d.SeasonalNaiveMAE]];
 overview.getRange(`H${row}`).formulas=[[`=(G${row}-D${row})/G${row}`]];
}
overview.getRange('A1:H16').format.columnWidth=19;overview.getRange('B:B').format.columnWidth=30;overview.getRange('D6:G7').setNumberFormat('0.00');overview.getRange('F6:F7').setNumberFormat('0.00"%"');overview.getRange('H6:H7').setNumberFormat('0.00%');overview.getRange('A6:H7').format.rowHeight=32;
const messages=[['Evaluation','All seven methods share 280 hourly / 43 daily May keys.'],['Selection','Hourly Historical Average; daily Random Forest.'],['Interpretation','Negative improvement means worse May MAE than Seasonal Naive.'],['May history','Previously examined during project development; not a fresh prospective test.'],['B reference','Stored Package B scores are separate from newly executed safe results.'],['Source','CSV tables and full grouped metrics are in 06_Model_Evaluation.']];
for(let i=0;i<messages.length;i++){const row=i+10;overview.getRange(`A${row}:B${row}`).merge();overview.getRange(`A${row}`).values=[[messages[i][0]]];overview.getRange(`C${row}:H${row}`).merge();overview.getRange(`C${row}`).values=[[messages[i][1]]];overview.getRange(`A${row}:H${row}`).format={wrapText:true,rowHeight:32,font:{name:'Aptos',size:11}};}
await wb.recalculate();
await fs.mkdir(path.join(root,'09_Documentation/Workbook_Previews'),{recursive:true});
for(const sh of [overview,...sheets.map(x=>x[0])]){
 const range=sh.name==='Requirements'?'A1:D8':sh.name==='Overview'?'A1:H15':`A1:${sh.name==='Daily approaches'?'F':sh.name==='A B New'?'G':'H'}12`;
 const blob=await wb.render({sheetName:sh.name,range,scale:1.4,format:'png'});await fs.writeFile(path.join(root,'09_Documentation/Workbook_Previews',sh.name.replaceAll(' ','_')+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(path.join(root,'06_Model_Evaluation/Overall_Model_Comparison.xlsx'));
console.log((await wb.inspect({kind:'region',sheetId:'Overview',range:'A5:H7',maxChars:1800,tableMaxRows:3,tableMaxCols:8})).ndjson);
console.log('Exported and rendered all ten workbook sheets.');
