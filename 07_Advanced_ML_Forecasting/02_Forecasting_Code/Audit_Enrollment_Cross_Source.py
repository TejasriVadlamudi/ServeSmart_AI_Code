"""Cross-check the supplied ML profile against existing timed Tableau evidence.
Audit only: no forecast feature, fit, selection or prediction is changed.
"""
from pathlib import Path
import pandas as pd,numpy as np,hashlib,json,shutil
R=Path(__file__).resolve().parents[1];src=Path.home()/'Desktop/ServeSmart_AI_FINAL_Dashboards/02_Data_Behind_Charts/Data_Behind_Dashboard4/Verified_Parsed_Building_Meetings.csv';dest=R/'01_Approved_Data/Source_Files/Timed_Schedule_Audit_Only';dest.mkdir(exist_ok=True)
if src.exists():shutil.copy2(src,dest/src.name)
m=pd.read_csv(dest/src.name);s=pd.read_csv(R/'01_Approved_Data/Weekly_Class_Profile_Source.csv');rows=[]
for r in s.itertuples():
 pool=m[m.Weekday.eq(r.DayOfWeek)];clock=pool[pool.StartMinute.le(r.Hour*60)&pool.EndMinute.gt(r.Hour*60)];overlap=pool[pool.StartMinute.lt((r.Hour+1)*60)&pool.EndMinute.gt(r.Hour*60)]
 rows.append(dict(Weekday=r.DayOfWeek,Hour=r.Hour,SuppliedClasses=r.ClassesActive,SuppliedEnrollments=r.EstimatedStudentsOnCampus,ParsedClockClasses=len(clock),ParsedClockEnrollments=clock.EstimatedEnrolled.sum(),ParsedOverlapClasses=len(overlap),ParsedOverlapEnrollments=overlap.EstimatedEnrolled.sum()))
x=pd.DataFrame(rows);x.to_csv(R/'06_Model_Evaluation/Enrollment_Timed_Evidence_Crosscheck.csv',index=False)
meta={'Status':'PARTIAL: existing timed evidence does not fully reconstruct the supplied ML profile under these two simple documented methods','Rows':len(x),'ClockClassRowsMatching':int(x.SuppliedClasses.eq(x.ParsedClockClasses).sum()),'ClockEnrollmentRowsMatching':int(x.SuppliedEnrollments.eq(x.ParsedClockEnrollments).sum()),'OverlapClassRowsMatching':int(x.SuppliedClasses.eq(x.ParsedOverlapClasses).sum()),'OverlapEnrollmentRowsMatching':int(x.SuppliedEnrollments.eq(x.ParsedOverlapEnrollments).sum()),'Source':str(src),'SourceSHA256':hashlib.sha256(src.read_bytes()).hexdigest(),'Caution':'This is a reference reconstruction with recovered timed meetings, not proof the supplied profile is wrong. Different term-date,section,deduplication or interval rules may explain differences. Obtain exact generator and effective-date/as-of metadata. No new training features were substituted.'}
(R/'06_Model_Evaluation/Enrollment_Cross_Source_Status.json').write_text(json.dumps(meta,indent=2));(dest/'READ_THIS_FIRST.txt').write_text('Audit-only copy of existing timed Tableau meeting evidence. It is not substituted into the approved ML file or new forecast features. See Enrollment_Timed_Evidence_Crosscheck.csv for comparisons and the explicit reconstruction caveat. The current master workbook still has blank section time columns; the exact supplied69-row profile generator remains unverified.\n')
print(json.dumps(meta,indent=2))
