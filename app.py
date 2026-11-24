import streamlit as st
import pandas as pd
import numpy as np
import time, hashlib, re
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(page_title="Smart Marketplace Analytics", page_icon="🛍️", layout="wide")
BASE = Path(__file__).parent
PRODUCT_FILE = BASE / "data" / "products.csv"
EVENT_FILE = BASE / "data" / "events.csv"

@st.cache_data
def load_products():
    return pd.read_csv(PRODUCT_FILE)
@st.cache_data
def load_events():
    e = pd.read_csv(EVENT_FILE)
    e["timestamp"] = pd.to_datetime(e["timestamp"])
    return e

def save_products(data):
    data.to_csv(PRODUCT_FILE, index=False)
    load_products.clear()

def version_key(data):
    return int(data["version"].max()) if len(data) else 0

@st.cache_data
def build_vector_index(texts, version):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=12000)
    matrix = vec.fit_transform(texts)
    return vec, matrix

def search_catalog(data, query="", category="All", min_price=0, max_price=100000, in_stock=False, method="Hybrid"):
    started=time.perf_counter()
    subset=data[(data.price>=min_price)&(data.price<=max_price)]
    if category!="All": subset=subset[subset.category==category]
    if in_stock: subset=subset[subset.stock>0]
    q=query.strip()
    if not q:
        result=subset.copy()
        result["score"]=1.0
    elif method=="Exact / substring":
        mask=(subset.title.str.contains(re.escape(q),case=False,na=False) |
              subset.description.str.contains(re.escape(q),case=False,na=False))
        result=subset[mask].copy()
        result["score"]=1.0
    else:
        corpus=(data.title.fillna("")+" "+data.description.fillna("")+" "+data.category.fillna("")).tolist()
        vec, matrix=build_vector_index(tuple(corpus), version_key(data))
        qv=vec.transform([q])
        all_scores=cosine_similarity(qv,matrix).ravel()
        score_map=dict(zip(data.product_id,all_scores))
        result=subset.copy()
        result["score"]=result.product_id.map(score_map).fillna(0)
        if method=="Keyword":
            tokens=set(re.findall(r"\w+",q.lower()))
            result["score"]=result.apply(lambda r: sum(t in (r.title+" "+r.description).lower() for t in tokens),axis=1)
        result=result[result.score>0].sort_values("score",ascending=False)
    elapsed=(time.perf_counter()-started)*1000
    return result, elapsed

products=load_products()
events=load_events()
categories=["All"]+sorted(products.category.unique().tolist())
st.title(" Smart Marketplace Search & Real-Time Analytics")
st.caption("Local prototype • Algorithm Engineering for Data Systems • Seeded demonstration data")
with st.sidebar:
    st.header("Workspace")
    st.write(f"Catalogue version: **{version_key(products)}**")
    st.write(f"Products: **{len(products):,}**")
    page=st.radio("Navigate",["Overview","Product Search","Catalogue Manager","Duplicate Detection","Visitor Analytics","Algorithm Lab"])
    st.divider()
    st.caption("Demo data is stored locally in data/products.csv and data/events.csv.")

if page=="Overview":
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Products",f"{len(products):,}")
    c2.metric("In stock",f"{(products.stock>0).sum():,}")
    c3.metric("Out of stock",f"{(products.stock==0).sum():,}")
    c4.metric("Catalogue version",version_key(products))
    st.subheader("Catalogue snapshot")
    a,b=st.columns(2)
    with a:
        cat=products.groupby("category").size().reset_index(name="products")
        st.plotly_chart(px.bar(cat,x="category",y="products",title="Products by category"),use_container_width=True)
    with b:
        st.plotly_chart(px.histogram(products,x="price",color="category",nbins=30,title="Price distribution"),use_container_width=True)
    st.subheader("Recent catalogue changes")
    st.dataframe(products.sort_values("updated_at",ascending=False).head(10),use_container_width=True,hide_index=True)
    st.info("Use Product Search to find items, Catalogue Manager to update stock, and Analytics for visitor activity.")

elif page=="Product Search":
    st.subheader("Search products")
    x1,x2,x3=st.columns([2,1,1])
    query=x1.text_input("Search query",placeholder="e.g. lightweight laptop")
    category=x2.selectbox("Category",categories)
    method=x3.selectbox("Search algorithm",["Hybrid","Exact / substring","Keyword"])
    y1,y2,y3=st.columns([1,1,1])
    min_price=y1.number_input("Minimum price (₹)",min_value=0,value=0,step=500)
    max_price=y2.number_input("Maximum price (₹)",min_value=0,value=100000,step=1000)
    in_stock=y3.checkbox("In-stock only",value=False)
    if max_price<min_price: st.error("Maximum price must be greater than or equal to minimum price.")
    else:
        result,elapsed=search_catalog(products,query,category,min_price,max_price,in_stock,method)
        st.caption(f"{len(result):,} results • {elapsed:.2f} ms measured search time • algorithm: {method}")
        st.dataframe(result[["product_id","title","category","price","stock","score","version"]].head(300),
                     use_container_width=True,hide_index=True)
        if len(result):
            st.download_button("Download results as CSV",result.to_csv(index=False),"search_results.csv","text/csv")
        else: st.warning("No products matched. Try a broader query or price range.")

elif page=="Catalogue Manager":
    st.subheader("Import, inspect and update inventory")
    uploaded=st.file_uploader("Import catalogue CSV (requires product_id, title, description, category, price, stock)",type="csv")
    if uploaded:
        incoming=pd.read_csv(uploaded)
        required={"product_id","title","description","category","price","stock"}
        if not required.issubset(incoming.columns):
            st.error("CSV is missing required columns: "+", ".join(sorted(required-set(incoming.columns))))
        elif st.button("Import catalogue"):
            incoming["version"]=version_key(products)+1
            incoming["updated_at"]=pd.Timestamp.now().isoformat()
            save_products(incoming)
            st.success(f"Imported {len(incoming)} products as catalogue version {version_key(incoming)}. Refresh the page.")
    st.markdown("**Update a product's stock and price**")
    pid=st.selectbox("Product ID",products.product_id.tolist())
    selected=products[products.product_id==pid].iloc[0]
    with st.form("update_product"):
        stock=st.number_input("New stock",min_value=0,value=int(selected.stock))
        price=st.number_input("New price (₹)",min_value=0,value=int(selected.price),step=100)
        submitted=st.form_submit_button("Save new version")
    if submitted:
        updated=products.copy()
        idx=updated.index[updated.product_id==pid][0]
        updated.loc[idx,"stock"]=stock
        updated.loc[idx,"price"]=price
        updated.loc[idx,"version"]=version_key(products)+1
        updated.loc[idx,"updated_at"]=pd.Timestamp.now().isoformat()
        save_products(updated)
        st.success(f"Saved {pid} at version {version_key(updated)}. Refresh the page to reload.")
    st.dataframe(products,use_container_width=True,hide_index=True)

elif page=="Duplicate Detection":
    st.subheader("Near-duplicate listing detection")
    threshold=st.slider("Similarity threshold",0.50,0.99,0.82,0.01)
    sample=products.head(500).copy()
    texts=(sample.title.fillna("")+" "+sample.description.fillna("")).tolist()
    vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),max_features=10000)
    mat=vec.fit_transform(texts)
    sim=cosine_similarity(mat)
    pairs=[]
    for i in range(len(sample)):
        for j in range(i+1,len(sample)):
            if sim[i,j]>=threshold:
                pairs.append({"Product A":sample.iloc[i].product_id,"Title A":sample.iloc[i].title,
                              "Product B":sample.iloc[j].product_id,"Title B":sample.iloc[j].title,
                              "Similarity":round(float(sim[i,j]),3)})
    st.metric("Potential duplicate pairs",len(pairs))
    if pairs: st.dataframe(pd.DataFrame(pairs).sort_values("Similarity",ascending=False),use_container_width=True,hide_index=True)
    else: st.success("No pairs found at this threshold.")
    st.caption("This is a text-similarity candidate detector. A manager should review candidates before merging listings.")

elif page=="Visitor Analytics":
    st.subheader("Visitor activity")
    window=st.selectbox("Time window",["24 hours","7 days","14 days"])
    days={"24 hours":1,"7 days":7,"14 days":14}[window]
    cutoff=pd.Timestamp.now()-pd.Timedelta(days=days)
    recent=events[events.timestamp>=cutoff].copy()
    c1,c2,c3=st.columns(3)
    c1.metric("Events",f"{len(recent):,}")
    c2.metric("Unique visitors",f"{recent.visitor_id.nunique():,}")
    c3.metric("Purchases",f"{(recent.event_type=='purchase').sum():,}")
    if len(recent):
        trend=recent.set_index("timestamp").resample("1h").size().reset_index(name="events")
        st.plotly_chart(px.line(trend,x="timestamp",y="events",title="Events over time (hourly)"),use_container_width=True)
        a,b=st.columns(2)
        with a: st.plotly_chart(px.pie(recent.groupby("event_type").size().reset_index(name="count"),names="event_type",values="count",title="Event mix"),use_container_width=True)
        with b:
            popular=recent.groupby("product_id").size().nlargest(10).reset_index(name="events")
            st.plotly_chart(px.bar(popular,x="product_id",y="events",title="Most active products"),use_container_width=True)
        st.dataframe(recent.sort_values("timestamp",ascending=False).head(100),use_container_width=True,hide_index=True)
    else: st.warning("No events in this time window.")

elif page=="Algorithm Lab":
    st.subheader("Search algorithm and scaling experiment")
    st.write("Compare exact substring scanning, keyword matching and TF-IDF hybrid retrieval on the same catalogue.")
    q=st.text_input("Experiment query", "lightweight laptop")
    sizes=st.multiselect("Catalogue sizes", [20,100,250,500,1000],default=[20,100,250,500,1000])
    if st.button("Run benchmark"):
        records=[]
        for n in sizes:
            data=products.head(n)
            for method in ["Exact / substring","Keyword","Hybrid"]:
                result,ms=search_catalog(data,q,"All",0,100000,False,method)
                records.append({"catalogue_size":n,"algorithm":method,"latency_ms":round(ms,3),"results":len(result)})
        bench=pd.DataFrame(records)
        st.dataframe(bench,use_container_width=True,hide_index=True)
        st.plotly_chart(px.line(bench,x="catalogue_size",y="latency_ms",color="algorithm",markers=True,title="Search latency vs catalogue size"),use_container_width=True)
        st.caption("Measured on this local run; repeat trials and report median/variance for formal evaluation.")
    st.divider()
    st.markdown("**External-memory block-read and shard/replica toy model**")
    block_size=st.slider("Records per block",10,100,50,10)
    shards=st.slider("Number of shards",1,8,4)
    workers=st.slider("Parallel workers",1,8,4)
    total=len(products)
    blocks=int(np.ceil(total/block_size))
    serial_reads=blocks
    parallel_work=int(np.ceil(blocks/min(shards,workers)))
    p1,p2,p3=st.columns(3)
    p1.metric("Total blocks",blocks)
    p2.metric("Serial block reads",serial_reads)
    p3.metric("Ideal parallel rounds",parallel_work)
    st.caption("Idealized model: assumes balanced shards, equal block cost and no network/coordination overhead. Real performance will differ.")

st.divider()
st.caption("Scope: local prototype with synthetic demo data. No live inventory, payment processing, external APIs or production deployment.")
