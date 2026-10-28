"""Reproducible descriptive analysis; export aggregates only, never patient rows."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/original/noshowappointments-kagglev2-may-2016.csv'
EXPECTED_SHA = '9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d'
LEAD = ['Same day', '1–7 days', '8–14 days', '15–30 days', '31+ days']
AGES = ['0–17', '18–34', '35–49', '50–64', '65+']
SEED, BOOTSTRAPS = 20261027, 600

def load_and_prepare():
    """Keep raw bytes intact; quarantine impossible dates/ages, retaining age extremes."""
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == EXPECTED_SHA
    raw = pd.read_csv(RAW, dtype={'PatientId': 'string', 'AppointmentID': 'string'})
    expected = ['PatientId','AppointmentID','Gender','ScheduledDay','AppointmentDay','Age',
                'Neighbourhood','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap',
                'SMS_received','No-show']
    assert raw.columns.tolist() == expected
    assert set(raw['No-show']) == {'No','Yes'}
    assert set(raw['SMS_received']) <= {0,1}
    assert raw.AppointmentID.notna().all() and raw.PatientId.notna().all()
    assert not raw.AppointmentID.duplicated().any(), 'Duplicate appointment IDs need manual adjudication'
    df = raw.copy()
    df['scheduled'] = pd.to_datetime(df.ScheduledDay, utc=True, errors='coerce')
    df['appointment'] = pd.to_datetime(df.AppointmentDay, utc=True, errors='coerce')
    # Calendar-day subtraction avoids labeling same-day bookings as negative due to midnight dates.
    df['lead_days'] = (df.appointment.dt.normalize() - df.scheduled.dt.normalize()).dt.days
    df['missed'] = df['No-show'].map({'No':0,'Yes':1}).astype('int8')
    invalid_age = df.Age.lt(0) | df.Age.isna()
    invalid_date = df.scheduled.isna() | df.appointment.isna() | df.lead_days.lt(0)
    keep = ~(invalid_age | invalid_date)
    audit = pd.DataFrame([
        ('Original appointments', len(raw), 'Immutable downloaded snapshot'),
        ('Exact duplicate rows', int(raw.duplicated().sum()), 'No automatic patient-level deduplication'),
        ('Missing source cells', int(raw.isna().sum().sum()), 'No imputation required if zero'),
        ('Invalid age rows', int(invalid_age.sum()), 'Exclude negative or absent age; retain zero and high ages'),
        ('Invalid date/lead rows', int(invalid_date.sum()), 'Exclude unparseable dates or negative calendar-day lead'),
        ('Overlapping exclusions', int((invalid_age & invalid_date).sum()), 'Count once in union'),
        ('Excluded unique rows', int((~keep).sum()), 'Union of age and date flags'),
        ('Analysis appointments', int(keep.sum()), 'Appointment-level analysis cohort'),
    ], columns=['check','count','decision'])
    clean = df.loc[keep].copy()
    clean['lead_group'] = pd.cut(clean.lead_days, [-1,0,7,14,30,np.inf], labels=LEAD)
    clean['age_group'] = pd.cut(clean.Age, [-1,17,34,49,64,np.inf], labels=AGES)
    clean['sms_group'] = clean.SMS_received.map({0:'No SMS recorded',1:'SMS recorded'})
    assert clean.lead_group.notna().all() and clean.age_group.notna().all()
    assert len(raw) == len(clean) + (~keep).sum()
    return raw, clean, audit

def summary(data, column):
    out = data.groupby(column, observed=True, sort=True).missed.agg(appointments='size', missed='sum')
    out['rate'] = out.missed / out.appointments
    return out.reset_index()

def clustered_intervals(data, column, reps=BOOTSTRAPS):
    """Percentile bootstrap resampling patients, retaining their full appointment clusters.

    Descriptive uncertainty only: observed patients are treated as exchangeable clusters.
    This does not correct unmeasured confounding, site/time dependence, or selection bias.
    """
    codes, patients = pd.factorize(data.PatientId, sort=True)
    groups = list(summary(data,column)[column])
    ns, ys = [], []
    for group in groups:
        select = (data[column] == group).to_numpy()
        ns.append(np.bincount(codes, weights=select.astype(float), minlength=len(patients)))
        ys.append(np.bincount(codes, weights=select * data.missed.to_numpy(), minlength=len(patients)))
    nmat, ymat = np.column_stack(ns), np.column_stack(ys)
    rng = np.random.default_rng(SEED)
    rates = np.empty((reps,len(groups)))
    for i in range(reps):
        sample = rng.integers(0,len(patients),len(patients))
        rates[i] = ymat[sample].sum(axis=0) / nmat[sample].sum(axis=0)
    out = summary(data,column)
    out['low_95'], out['high_95'] = np.quantile(rates,[.025,.975],axis=0)
    return out

def make_charts(lead, age, sms_lead, cohort):
    figures = ROOT / 'results/figures'
    figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':150})
    def save(fig, name):
        fig.savefig(figures / name, dpi=170, bbox_inches='tight',facecolor='white')
        plt.close(fig)
    for table,col,title,filename in [
        (lead,'lead_group','Missed appointments rise with scheduling lead time','01-lead-time.png'),
        (age,'age_group','Attendance varies across broad age groups','02-age-groups.png')]:
        fig,ax=plt.subplots(figsize=(10,5.3))
        x=np.arange(len(table))
        ax.bar(x,table.rate,color='#007f86',width=.62)
        ax.errorbar(x,table.rate,yerr=[table.rate-table.low_95,table.high_95-table.rate],
                    fmt='none',color='#20313f',capsize=4)
        for j,r in table.iterrows():
            ax.text(j,r.high_95+.013,f'{r.rate:.1%}\nn={int(r.appointments):,}',ha='center',fontsize=10)
        ax.set(xticks=x,xticklabels=table[col].astype(str),ylim=(0,max(.45,table.high_95.max()+.1)),
               ylabel='Missed appointments / all appointments',title=title,
               xlabel='Calendar days from booking to appointment' if col=='lead_group' else 'Recorded age (years)')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        fig.text(.12,.01,'Bars are descriptive; whiskers: 95% patient-cluster bootstrap intervals (600 replicates).',fontsize=9,color='#52606d')
        fig.tight_layout(rect=(0,.05,1,1)); save(fig,filename)
    fig,ax=plt.subplots(figsize=(10,5.4))
    # Same-day SMS group has zero observations; excluded from this comparison by design.
    for value,label,color in [(0,'No SMS recorded','#007f86'),(1,'SMS recorded','#d47b25')]:
        part=sms_lead[(sms_lead.SMS_received==value)&(sms_lead.lead_group!='Same day')]
        ax.plot(np.arange(len(part)),part.rate,'o-',label=label,color=color,lw=2)
        for x,r in enumerate(part.itertuples()):
            ax.annotate(f'n={r.appointments:,}',(x,r.rate),xytext=(0,9 if value==0 else -17),
                        textcoords='offset points',ha='center',fontsize=8)
    ax.set(xticks=np.arange(4),xticklabels=LEAD[1:],ylim=(0,.5),ylabel='Missed appointment rate',
           xlabel='Scheduling lead time',title='Reminder comparisons change when lead time is considered')
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(frameon=False)
    fig.text(.12,.01,'Appointments booked at least one day ahead only. Observational comparisons do not estimate SMS effects.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'03-sms-within-lead.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,column,title in zip(axes,['age','lead_days'],['Age distribution','Scheduling lead distribution']):
        s=cohort.loc[cohort.variable==column].iloc[0]
        ax.bxp([{'med':s['50%'],'q1':s['25%'],'q3':s['75%'],'whislo':s['min'],'whishi':s['max'],'fliers':[]}],showfliers=False,vert=False)
        ax.set(title=title,xlabel='Years' if column=='age' else 'Calendar days',yticks=[])
    fig.text(.08,.01,'Aggregate five-number summaries. Whiskers show minimum and maximum, not the 1.5×IQR rule.',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,'04-distributions.png')

def run_analysis():
    raw,clean,audit=load_and_prepare()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    lead=clustered_intervals(clean,'lead_group')
    age=clustered_intervals(clean,'age_group')
    sms=summary(clean,'sms_group')
    sms_lead=clean.groupby(['lead_group','SMS_received'],observed=True).missed.agg(appointments='size',missed='sum').reset_index()
    sms_lead['rate']=sms_lead.missed/sms_lead.appointments
    cohort=clean[['Age','lead_days']].rename(columns={'Age':'age'}).describe().T.reset_index(names='variable')
    # Use pooled stratum weights on common-support lead bands for a descriptive adjusted comparison.
    pos=sms_lead[sms_lead.lead_group!='Same day']
    piv=pos.pivot(index='lead_group',columns='SMS_received',values='rate')
    weights=pos.groupby('lead_group',observed=True).appointments.sum()
    weights=weights/weights.sum()
    std={int(k):float((piv[k]*weights).sum()) for k in [0,1]}
    earliest=clean.sort_values(['scheduled','appointment','AppointmentID'],kind='stable').drop_duplicates('PatientId',keep='first')
    sensitivities=pd.DataFrame([
        ('All valid appointments',len(clean),clean.missed.mean()),
        ('One earliest-booked valid appointment per patient',len(earliest),earliest.missed.mean()),
        ('Valid appointments, age at most 100',int(clean.Age.le(100).sum()),clean.loc[clean.Age.le(100),'missed'].mean()),
        ('Original records with recognized outcome',len(raw),raw['No-show'].eq('Yes').mean()),
    ],columns=['cohort','appointments','rate'])
    first_lead=summary(earliest,'lead_group')
    overview={
        'raw_appointments':len(raw),'analysis_appointments':len(clean),'excluded':len(raw)-len(clean),
        'unique_patients':int(clean.PatientId.nunique()),'missed':int(clean.missed.sum()),
        'missed_rate':float(clean.missed.mean()),'appointment_start':str(clean.appointment.min().date()),
        'appointment_end':str(clean.appointment.max().date()),'seed':SEED,'bootstrap_replicates':BOOTSTRAPS,
        'standardized_sms_rate':std[1],'standardized_no_sms_rate':std[0],
        'standardized_difference_pp':100*(std[1]-std[0]),
        'raw_sha256':EXPECTED_SHA,
    }
    # Suppress rows with fewer than 100 appointments from public grouped outputs.
    # Cell-level suppression is conservative portfolio hygiene, not a formal privacy guarantee.
    public_sms=sms_lead[(sms_lead.appointments>=100)&(sms_lead.lead_group!='Same day')].copy()
    tables={'cleaning-audit':audit,'lead-time':lead,'age-groups':age,'sms-overall':sms,
            'sms-by-lead':public_sms,'descriptive-statistics':cohort,'sensitivity':sensitivities,
            'first-appointment-lead':first_lead}
    for name,table in tables.items():
        table.to_csv(out/f'{name}.csv',index=False,float_format='%.8f')
    (out/'summary.json').write_text(json.dumps(overview,indent=2)+'\n',encoding='utf-8')
    make_charts(lead,age,public_sms,cohort)
    return overview,tables

if __name__=='__main__':
    overview,_=run_analysis()
    print(json.dumps(overview,indent=2))
