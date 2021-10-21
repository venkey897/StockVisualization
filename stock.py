import sys
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web
import dash
from dash.dependencies import Input,Output
from dash import dcc
from dash  import  html
from datetime import datetime
import yfinance as yf
import cgi
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
app = dash.Dash(__name__)
server = app.server
app.layout=html.Div(
children=[html.H1("Welcome to stock prediction"),
    html.Div(
dcc.Input(id="stack",value=" ",type="text")),
dcc.DatePickerSingle(id='date', className='hello',
    date=datetime(2015,3,12)
),
html.Br(),
html.Br(),
html.Div(id="out",style={'textAlign':'center','color':'#111111'}),
html.Div(id="intro",style={'textAlign':'center','color':'#111111'}),
html.Div(id='vol',style={'textAlign':'center','color':'#111111'}),
dcc.Input(id="name",value='',type='text')
])
@app.callback(
    Output(component_id="out",component_property='children'),
    Output(component_id="intro",component_property='children'),
    Output(component_id="vol",component_property='children'),
    Input(component_id="stack",component_property="value"),
    Input(component_id="date",component_property="date"))
def update_val(inp,date):
    if inp is None:
        return render_template("file:///C:/Users/VENKEY/Documents/hello.html")
    else:
        datare=yf.Ticker(inp)
        inf=datare.info
        d=pd.DataFrame().from_dict(inf,orient="index").T
        dz=yf.Ticker(inp)
        inf=dz.info
        d=pd.DataFrame().from_dict(inf,orient="index").T
        d=d[["logo_url","shortName","longBusinessSummary"]]
        k=d['longBusinessSummary'].values[0]
        end=datetime(2020,10,12)
        try:
            df=web.DataReader(inp,'yahoo',date,end)
            fig=dcc.Graph(id="demo",figure={'data':[{'x':df.index,'y':df.Close,'type':'line','name':inp},],'layout':{'title':inp}})
            model=tf.keras.models.load_model("hello.model")
            new=df.filter(['Close'])
            scaler=MinMaxScaler(feature_range=(0,1))
            l=new[-60:].values
            l=scaler.fit_transform(l)
            l=scaler.transform(l)
            x_test=[]
            x_test.append(l)
            x_test=np.array(x_test)
            x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
            pre=model.predict(x_test)
            pre=scaler.inverse_transform(pre)
            pre="Tommarrow may be"+str(pre)
            ploted=html.Table([html.Tr([
                html.Thead(
                    html.Tr([html.Th(col) for col in df.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(df.iloc[i][col] )for col in df.columns
                    ])for i in range(min(len(df),20))
                ])
            ])])
            return  k,fig,ploted
        except AttributeError:
            pass

if __name__ == '__main__':
    app.run_server(debug=True)
