"""V19.0 mobile-first research dashboard. Every empty field has an explicit state."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import html
import json
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from v19_config import Settings, FACTORS, TECH, TZ, VERSION
from v19_data import read_json
from v19_features import current_candidates
from v19_service import (start_job,job_status,material_for_display,read_last_report,
                         weights_for,analog_summary,field_health,position_size)

ROOT=Path(__file__).resolve().parent
BASE=Path(os.getenv('ALPHA_RADAR_DATA_DIR',str(ROOT/'data')))
DATA=BASE/'v19'
LABELS={
 'trend':'趨勢／均線','relative_strength':'相對大盤強度','breakout':'平台突破',
 'reversal':'KD／MACD／RSI修復','pattern':'W底／頸線','volume':'價量搭配',
 'revenue':'營收成長','quality':'獲利品質','foreign':'外資持續買超','trust':'投信持續買超',
 'joint':'外投同步買超','dealer':'自營商自行買賣','margin':'融資整理','holders':'大戶持股變化',
 'ticker':'股票代碼','name':'名稱','date':'資料日期','close':'日線收盤','source':'行情來源',
 'rsi':'RSI(14)','k':'KD：K值','d':'KD：D值','macd_hist':'MACD柱體','bias20':'20日乖離％',
 'neckline':'W底頸線','real_market_validation':'10點歷史研究'}
SETUP={'REVERSAL':'低檔修復觀察','BREAKOUT':'突破加速觀察','GROWTH_PULLBACK':'業績＋回檔觀察','WATCH':'趨勢追蹤'}
FIT={'STARTING_PRIOR_NO_FIT':'起始權重，尚未訓練','BASELINE_INSUFFICIENT_TRAINING':'訓練樣本不足，沿用基準',
     'BASELINE_TECH_GAP':'技術資料不足，沿用基準','BASELINE_VALIDATION_NOT_BETTER':'內部驗證未明顯勝出，保留基準',
     'RESEARCH_FIT':'已依當時以前資料調整（研究用途）'}
CSS="""
<style>
.block-container{max-width:1120px;padding-top:1.1rem;padding-bottom:3rem}
h1{font-size:1.85rem!important;letter-spacing:-.025em}h2{font-size:1.3rem!important}
.v19-note{border-left:4px solid #537cb6;background:#f1f5fb;padding:12px 16px;border-radius:6px;
color:#304663;margin:12px 0;line-height:1.7}
.v19-card{border:1px solid #dbe3ef;border-radius:16px;padding:18px 20px;margin:13px 0;background:#fff;
box-shadow:0 3px 12px rgba(38,60,90,.04);color:#20334c}
.v19-title{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:1.13rem;font-weight:750}
.v19-tag{display:inline-block;background:#ecf3fc;color:#395f90;padding:4px 9px;border-radius:7px;
font-size:.83rem;font-weight:600;margin:9px 6px 5px 0}
.v19-small{font-size:.88rem;color:#647287;line-height:1.65;overflow-wrap:anywhere}
.v19-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:12px 0}
.v19-metrics span{display:block;font-size:.78rem;color:#647287}.v19-metrics b{font-size:1.22rem}
.v19-up{color:#bc4740}.v19-down{color:#17795c}
.v19-muted{color:#68788c}
[data-testid="stRadio"] [role="radiogroup"]{flex-wrap:wrap;gap:5px}
[data-testid="stDataFrame"]{max-width:100%}
@media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem}
.v19-card{padding:15px 13px}.v19-title{font-size:1.05rem}.v19-metrics{gap:8px}
.v19-metrics b{font-size:1.07rem}.v19-metrics span{font-size:.75rem}}
</style>
"""

def esc(x):return html.escape(str(x if x is not None else ''),quote=True)

def number(value,digits=2):
    try:
        v=float(value)
        return f'{v:,.{digits}f}' if np.isfinite(v) else '—'
    except (TypeError,ValueError):return '—'

def percent(value,scale=100):
    try:
        v=float(value)*scale
        return f'{v:+.2f}%' if np.isfinite(v) else '—'
    except (TypeError,ValueError):return '—'

def _bootstrap():
    if not os.getenv('FINMIND_TOKEN'):
        try:
            token=st.secrets.get('FINMIND_TOKEN')
            if token:os.environ['FINMIND_TOKEN']=str(token)
        except Exception:pass

@st.cache_data(ttl=600,show_spinner=False)
def _display_data(root,price_mtime,evidence_mtime,legacy_mtime,cfg):
    return material_for_display(root,cfg)

def _mtime(path):
    try:return Path(path).stat().st_mtime
    except OSError:return 0.

@st.fragment(run_every=3)
def progress_box():
    status=job_status(DATA)
    if not status:return
    state=status.get('state')
    stages={'prepare':'準備','download':'下載歷史日線','optional':'補充營收／籌碼','features':'計算因子','validate':'歷史驗證'}
    stage=stages.get(status.get('stage'),status.get('stage',''))
    if state=='RUNNING':
        total=max(1,int(status.get('total',1)));done=int(status.get('done',0))
        st.progress(min(1.,done/total),text=f"{stage}：{done}/{total}｜{status.get('message','')}")
        st.caption('可先查看其他頁面。手機重新連線不會重開同一個任務；伺服器重啟後可從已下載資料接續。')
    elif state=='DONE':
        st.success('本次流程完成：'+str(status.get('message','')))
        key='v19_seen_'+str(status.get('run_id',''))
        if not st.session_state.get(key):
            st.session_state[key]=True
            _display_data.clear()
            st.rerun()
    elif state in ('FAILED','INTERRUPTED'):
        st.warning('本次更新未完成；先前成功結果仍保留。'+str(status.get('message','')))
    with st.expander('執行紀錄',expanded=False):
        st.json(status)

def candidate_card(row,analog,horizon,weights,index):
    valid=analog.get('state')=='HISTORICAL_ANALOG_NOT_CALIBRATED_FORECAST'
    median=analog.get('median_net') if valid else None
    mean=analog.get('mean_net') if valid else None
    count=analog.get('count',0)
    tone='v19-up' if median is not None and median>0 else 'v19-down' if median is not None else 'v19-muted'
    factors=sorted(((k,float(row.get(k,np.nan))*v) for k,v in weights.items()
                   if pd.notna(row.get(k))),key=lambda x:x[1],reverse=True)[:3]
    reasons='／'.join(LABELS.get(k,k) for k,_ in factors) or '因子資料不足'
    risk=number(row.get('stop_distance_pct'))
    missing=[LABELS[k] for k in FACTORS if pd.isna(row.get(k))]
    evidence='可比較的歷史樣本不足' if not valid else f"樣本 {count} 組｜歷史正報酬比例 {percent(analog.get('positive_fraction'))}"
    st.markdown(f"""<div class="v19-card">
      <div class="v19-title"><div>#{index:02d}　{esc(row.get('name'))} <span class="v19-small">{esc(row['ticker'])}</span></div>
        <div>日線 {number(row.get('close'))}</div></div>
      <span class="v19-tag">{esc(SETUP.get(row.get('setup'),'研究觀察'))}</span>
      <span class="v19-tag">研究分 {number(row.get('score'),1)}</span>
      <span class="v19-tag">因子覆蓋 {number(float(row.get('factor_coverage',0))*100,0)}%</span>
      <div class="v19-metrics">
        <div><span>{horizon}日歷史平均淨報酬</span><b>{percent(mean)}</b></div>
        <div><span>歷史中位數（非保證）</span><b class="{tone}">{percent(median)}</b></div>
        <div><span>參考停損距離</span><b>{risk}%</b></div>
      </div>
      <div class="v19-small">{esc(reasons)}<br>{esc(evidence)}<br>
      觀察區 {number(row.get('reference_low'))}–{number(row.get('reference_high'))}｜
      防守參考 {number(row.get('stop_reference'))}｜不追價參考 {number(row.get('chase_ceiling'))}<br>
      資料日 {esc(str(row.get('date'))[:10])}｜來源 {esc(row.get('source'))}<br>
      這是日線研究候選，不是即時買進指令；價區以本次日線與ATR估算。</div></div>""",unsafe_allow_html=True)
    if missing:
        with st.expander(f"{row['ticker']}：因子數值與缺資料說明",expanded=False):
            table=[{'因子':LABELS.get(k,k),'分數':number(row.get(k),1),
                    '本次權重':f"{weights.get(k,0)*100:.1f}%",
                    '狀態':'缺資料，不填0' if pd.isna(row.get(k)) else '可用'} for k in FACTORS]
            st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)

def page_decisions(panel,report,cfg):
    st.subheader('先看機會，再看進場，最後控制投入金額')
    st.markdown('<div class="v19-note">排序、買點與資料可信度分開。研究分高不代表一定漲停；歷史平均與中位數不是未來報酬保證。沒有營收／獲利證據的股票，不會冒稱「業績優良」。</div>',unsafe_allow_html=True)
    if panel.empty:
        st.info('尚未載入行情。按上方「下載近年資料＋執行驗證」即可；已有V18資料庫時會先讀取，無須清除原資料。')
        return
    h=st.radio('持有期（交易日）',list(cfg.horizons),horizontal=True,format_func=lambda x:f'{x}日',key='v19_h')
    asof=panel.date.max();weights,state=weights_for(report,h,asof)
    st.caption(f"日線截至 {str(asof.date())}｜{FIT.get(state,state)}。來源標示與日期分開，不把來源待核一律叫舊日線。")
    rows=current_candidates(panel,cfg,weights,h)
    pool_counts=panel[panel.date==asof]
    st.caption(f"本日有資料 {pool_counts.ticker.nunique()} 檔；通過歷史長度／流動性／價格異常檢查 {int(pool_counts.eligible.sum())} 檔。這是已載入股票池，不冒稱全市場。")
    mode=st.selectbox('機會類型',['全部研究候選','業績＋回檔','低檔修復','突破加速'],key='v19_setup')
    codes={'業績＋回檔':'GROWTH_PULLBACK','低檔修復':'REVERSAL','突破加速':'BREAKOUT'}
    if mode in codes:rows=rows[rows.setup==codes[mode]]
    if rows.empty:
        st.info('這個類型目前沒有通過條件的候選。可能是條件未成立或基本面證據不足，不等於市場沒有機會。')
    for i,(_,row) in enumerate(rows.head(5).iterrows(),1):
        candidate_card(row,analog_summary(panel,row,h,weights),h,weights,i)
    with st.expander('本次因子權重與使用方式',expanded=False):
        st.dataframe(pd.DataFrame([{'因子':LABELS.get(k,k),'權重％':round(v*100,2)} for k,v in weights.items()]),hide_index=True,use_container_width=True)
        st.write('一、研究分是相對排序，不是上漲機率。二、確認型態、量價與當日資料後才評估進場。三、用預定損失設定部位；停損不保證跳空時能成交。四、每次研究使用相同成本與股票池，與固定權重及0050比較。')
    with st.expander('資金配置試算（不是自動下單）',expanded=False):
        capital=st.number_input('可投入資金（元）',min_value=1000.,value=100000.,step=10000.,key='v19_capital')
        risk=st.slider('每筆規劃損失／資金（％）',.1,2.,.5,.1,key='v19_risk')
        entry=st.number_input('預計買進價',min_value=.01,value=100.,key='v19_entry')
        stop=st.number_input('預定防守價',min_value=.01,value=95.,key='v19_stop')
        sized=position_size(capital,entry,stop,risk/100)
        if sized['shares']:
            st.write(f"參考上限：{sized['shares']:,} 股；部位約 {number(sized['position_value'],0)} 元；含成本緩衝的規劃損失約 {number(sized['planned_max_loss'],0)} 元。")
        else:st.info('請確認防守價低於買進價，且輸入值合理。')
        st.caption('單股上限20％，可含零股；未處理實際最低手續費。跳空、流動性及滑價可能使損失超過規劃。')

def page_validation(panel,report,cfg):
    st.subheader('10個基準日：先封存選股，再看後續表現')
    st.write('從近年資料分段隨機抽取10個基準日，固定種子1900。當次權重只使用該日以前已完成的績效；先篩選訓練候選，再以更晚的內部驗證區段選權重。')
    st.caption('持有期為交易日：基準日T計算 → T+1開盤假設進場 → T+5／20／60收盤評價。同步匯出相對T日收盤的價格變動。未成交名額留現金，不以其他成功股票替補。')
    if not report:
        st.info('尚無已完成研究。可按「下載近年資料＋執行驗證」，或用已有快取執行。缺籌碼不會阻止價量因子的驗證。')
        return
    st.write(f"研究狀態：**{report.get('status')}**｜資料 {report.get('data_start')}～{report.get('data_end')}｜股票／基準合計 {report.get('symbols')} 檔")
    st.caption('目前資料池未完整驗證歷史下市成分；仍有存活者偏差。若來源為未還原價格，結果是扣成本的價格報酬，不等同含息總報酬。')
    kind=report.get('data_kind','UNCLASSIFIED')
    st.caption('資料分類：'+kind+'；RESEARCH_COMPLETE只表示本次資料的計算流程完成，不代表最佳策略已獲證實。')
    if kind=='SYNTHETIC_TEST_ONLY':
        st.warning('此為合成測試資料，不能當作真實市場績效。')
    counts=report.get('fit_status_counts') or {}
    if counts:
        with st.expander('各次權重調整結果',expanded=False):
            st.json(counts)
    records=pd.DataFrame(report.get('results',[]))
    if not records.empty:
        table=records.pivot_table(index=['date','horizon'],columns='strategy',values='mean_net',aggfunc='first',dropna=False).reset_index()
        table=table.rename(columns={'date':'基準日','horizon':'持有交易日','baseline':'固定權重','adaptive':'逐步調整權重',
                                    'benchmark_0050':'0050同口徑','equal_weight_universe':'股票池等權'})
        for c in ('固定權重','逐步調整權重','0050同口徑','股票池等權'):
            if c in table:table[c]=table[c].map(percent)
        st.dataframe(table,hide_index=True,use_container_width=True)
        st.caption('「—」代表該組不完整，不是報酬0。60日結果可能跨除權息、分割或停牌；異常未還原價格區間不拿來證明策略勝出。')
        summary=records[records.strategy.isin(['baseline','adaptive'])].groupby(['horizon','strategy']).agg(
            mean_net=('mean_net','mean'),complete_cases=('complete','sum')).reset_index()
        st.dataframe(summary.rename(columns={'horizon':'持有交易日','strategy':'方案','mean_net':'平均淨報酬（小數）','complete_cases':'完整基準日數'}),hide_index=True,use_container_width=True)
    with st.expander('每次訓練、排除因子與完整來源紀錄',expanded=False):
        st.json({'fits':report.get('fits'), 'overlap':report.get('overlapping_horizons'),
                 'adjusted_bar_fraction':report.get('adjusted_bar_fraction'),'manifest':report.get('manifest')})
    pointer=read_json(DATA/'last_success.json')
    directory=Path(pointer['directory']) if pointer.get('directory') else None
    if directory and directory.resolve().is_relative_to((DATA/'runs').resolve()):
        for fn,label,mime in [('comparison.csv','下載10個基準日比較表','text/csv'),
                              ('selected_trades.csv','下載每檔選股／買賣基準／結果','text/csv'),
                              ('v19_results.json','下載完整結果與每輪權重','application/json'),
                              ('v19_validation.md','下載研究報告','text/markdown')]:
            path=directory/fn
            if path.exists():st.download_button(label,path.read_bytes(),file_name=fn,mime=mime,key='dl_'+fn)
    with st.expander('逐欄位檢查（全部已載入股票）',expanded=False):
        health=pd.DataFrame(field_health(panel,report))
        if 'field' in health:health['field']=health.field.map(lambda x:LABELS.get(x,x))
        st.dataframe(health.rename(columns={'field':'欄位','present':'有值筆數','total':'母數','coverage':'有值比例','state':'狀態'}),hide_index=True,use_container_width=True)
        st.caption('有值比例不等於資料正確率；頸線空值也可能表示沒有W底。營收、法人、大戶等缺值不會補成0或捏造連買日數。')

@st.fragment(run_every=20)
def live_tile(ticker):
    from v19_live import observation
    quote=observation(ticker)
    if quote.get('pending') and quote.get('price') is None:
        st.info('正在查詢當日行情；日線分析仍可使用，不需要等待整頁重新計算。')
    if quote.get('price') is not None:
        st.metric('當日觀察價',number(quote['price']),delta=percent(quote.get('change_pct'),1))
        st.caption(f"來源 {quote.get('source','—')}｜成交／分鐘時間 {quote.get('timestamp','—')}｜{quote.get('age_label','—')}")
        if quote.get('observed_black_to_red'):st.info('本次追蹤已觀察到由昨收下方轉至上方；這是已發生的現象，不是事前預測命中。')
    else:
        st.warning('當日成交價尚未取得：'+str(quote.get('reason') or '來源仍查詢中'))
    st.caption('盤中與日線日期分開。第三方分鐘價、過期官方價不冒充可成交買點；V19新權重尚不自動發布下單指令。')
    with st.expander('盤中來源診斷',expanded=False):
        st.json(quote)

def page_doctor(panel,report,cfg):
    st.subheader('個股診斷：日線結構與當日行情分開')
    if panel.empty:
        st.info('請先下載歷史日線，或使用原有V18資料庫。')
        return
    names=panel.sort_values('date').drop_duplicates('ticker',keep='last').set_index('ticker')['name'].to_dict()
    tickers=sorted(names)
    default=tickers.index('2330.TW') if '2330.TW' in tickers else 0
    ticker=st.selectbox('已載入股票',tickers,index=default,format_func=lambda t:f"{names[t]} {t}",key='v19_doctor_ticker')
    h=st.radio('評估週期',list(cfg.horizons),horizontal=True,format_func=lambda x:f'{x}交易日',key='v19_doctor_h')
    stock=panel[panel.ticker==ticker].sort_values('date');row=stock.iloc[-1]
    weights,state=weights_for(report,h,row.date)
    # Display does not require a stock to pass the candidate filter.
    from v19_features import score_rows
    scored=stock.tail(1).copy();scored['score'],scored['factor_coverage']=score_rows(scored,weights)
    observed=scored.iloc[-1].to_dict()
    observed['setup']='WATCH';observed['stop_reference']=max(float(row.support10),float(row.close-1.8*row.atr)) if pd.notna(row.atr) and pd.notna(row.support10) else np.nan
    observed['stop_reference']=min(observed['stop_reference'],float(row.close)*.99)
    observed['stop_distance_pct']=(row.close-observed['stop_reference'])/row.close*100
    observed['reference_low']=row.close-.5*row.atr;observed['reference_high']=row.close+.25*row.atr
    observed['chase_ceiling']=row.close+min(row.atr,row.close*.03)
    candidate_card(observed,analog_summary(panel,row,h,weights),h,weights,1)
    st.caption(f"股票最新日線 {str(row.date.date())}；整體模型最新資料 {str(panel.date.max().date())}。")
    if st.checkbox('顯示當日盤中觀察（每20秒檢查）',value=False,key='v19_live_enable'):
        live_tile(ticker)
    data=stock.tail(120)
    fig=go.Figure(go.Candlestick(x=data.date,open=data.open,high=data.high,low=data.low,close=data.close,name='日K'))
    fig.add_trace(go.Scatter(x=data.date,y=data.ma20,name='20日均線'))
    fig.add_trace(go.Scatter(x=data.date,y=data.ma60,name='60日均線'))
    fig.update_layout(height=390,margin=dict(l=8,r=8,t=20,b=10),xaxis_rangeslider_visible=False,legend=dict(orientation='h'))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(pd.DataFrame([{'指標':LABELS.get(k,k),'數值':number(row.get(k))} for k in ('rsi','k','d','macd_hist','bias20','neckline')]),hide_index=True,use_container_width=True)
    if not bool(row.eligible):st.warning('本檔未通過目前候選池條件；可查看資料，不會自動當成推薦股。')

def news_block():
    with st.expander('可能影響台股：最近72小時新聞與國際摘要',expanded=False):
        from briefing_jobs import load_brief,ensure_brief
        from market_briefing import _only_recent_news
        if st.button('更新新聞／國際摘要',key='v19_news_refresh'):
            ensure_brief(BASE,force=True)
            st.info('已啟動新聞更新；稍後重新開啟此區即可查看。')
        brief=load_brief(BASE);news=_only_recent_news(brief.get('news') or {},now=datetime.now(TZ),max_age_hours=72)
        if not news.get('items'):st.caption('目前沒有可核對發布時間的72小時內新聞；不以舊新聞補滿。')
        for row in (news.get('items') or [])[:10]:
            st.markdown('**'+str(row.get('title',''))+'**')
            st.caption(str(row.get('published_iso',''))+'｜'+str(row.get('source','')))
            if row.get('summary'):st.write(row['summary'])
            if row.get('impact'):st.write(row['impact'])
            if str(row.get('link','')).startswith(('https://','http://')):st.link_button('來源文章',row['link'])

def main():
    st.set_page_config(page_title='Alpha Radar V19.0',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
    _bootstrap();st.markdown(CSS,unsafe_allow_html=True)
    st.title('Alpha Radar V19.0')
    st.caption('近年日線驗證 × 可追溯權重 × 投資規劃｜研究版，非保證獲利或自動交易')
    with st.sidebar:
        st.subheader('資料與測試設定')
        years=st.selectbox('歷史區間（年）',[1,2,3],index=2,key='v19_years')
        n=st.selectbox('股票池上限',[20,50,100,200],index=2,key='v19_stocks')
        end=st.date_input('資料截止日',value=datetime.now(TZ).date()-timedelta(days=1),
                          max_value=datetime.now(TZ).date()-timedelta(days=1),key='v19_end')
        extra=st.checkbox('另抓營收／法人／融資（較耗額度）',value=False,key='v19_optional')
        st.caption('預設先完成价量驗證。選配資料缺漏不阻擋主流程；可分批接續，無須先購買資料服務。')
        legacy=st.checkbox('開啟V18.21原工作台（保留舊模型）',value=False,key='v19_legacy')
    if legacy:
        st.info('以下是保留的V18.21原工作台，不是V19新研究排名。')
        import legacy_app
        legacy_app.main(configure_page=False)
        return
    cfg=Settings(years=years,stocks=n,as_of=end.isoformat(),optional_data=extra)
    c1,c2=st.columns(2)
    with c1:
        if st.button('下載近年資料＋執行10點驗證',type='primary',key='v19_start'):
            st.info(str(start_job(DATA,cfg,'download_validate').get('state')))
    with c2:
        if st.button('只用既有快取重新驗證',key='v19_cached'):
            st.info(str(start_job(DATA,cfg,'cache_validate').get('state')))
    progress_box()
    try:
        panel,_=_display_data(str(DATA),_mtime(DATA/'history_prices.csv.gz'),
                             _mtime(DATA/'historical_evidence.csv.gz'),_mtime(BASE/'daily_prices.sqlite'),cfg)
        report=read_last_report(DATA)
    except Exception as exc:
        panel=pd.DataFrame();report=read_last_report(DATA)
        st.error('資料檢查未通過，未填入假數字：'+type(exc).__name__+': '+str(exc))
    view=st.radio('功能',['選股與規劃','歷史驗證','個股診斷'],horizontal=True,label_visibility='collapsed',key='v19_view')
    if view=='選股與規劃':page_decisions(panel,report,cfg);news_block()
    elif view=='歷史驗證':page_validation(panel,report,cfg)
    else:page_doctor(panel,report,cfg)
    st.caption('公開日線可以先驗證技術策略；歷史營收、籌碼與大戶資料只有在發布時間可用時才加入。10個基準日是診斷，不足以證明全域最佳權重。')
