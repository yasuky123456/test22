#https://gri.jp/media/entry/3798
#回帰分析評価指標
#https://di-acc2.com/analytics/ai/8852/
#全体のまとめ
#https://qiita.com/shuhigashi/items/0fb37468e64c76f4b245
from cgi import test
import streamlit as st
import pandas as pd
import datetime
#from pycaret.classification import load_model, predict_model
from pycaret.regression import load_model, predict_model
import pickle
import streamlit.components.v1 as stc
from PIL import Image

image2 = Image.open('pycaret.png')
st.image(image2)
st.write('')



st.markdown('**参考記事**.')
st.markdown("[※はじめに:予測モデルの流れ．](https://plazma.red/csblog-machine-learning/)")
st.markdown("[※解説1:分類？回帰？違いとは．](https://avinton.com/academy/classification-regression/)")
st.markdown("[※解説2: cross-validation](https://toukei-lab.com/over-fitting)")
st.markdown("[※解説3: ハイパーパラメータと最適化](https://www.codexa.net/hyperparameter-tuning-python/)")
st.markdown("[※解説4: グラフ解説](https://qiita.com/ground0state/items/57e565b23770e5a323e9)")
st.markdown("[※解説5: 分類モデルの性能評価？評価指標？](https://di-acc2.com/analytics/ai/6089/)")
st.markdown("[※解説6: pycaret(分類モデル)のまとめ←おススメ！](https://qiita.com/shuhigashi/items/cb6816a1da1d347bbdc2)")
st.markdown("[※解説7: 回帰モデルの性能評価？評価指標？](https://di-acc2.com/analytics/ai/8852/)")
st.markdown("[※解説8: pycaret(回帰モデル)のまとめ←おススメ！](https://qiita.com/shuhigashi/items/0fb37468e64c76f4b245)")

st.markdown('**key word**.')
"""
python.pycaret. jupyter notebook.  Google Colab.  streamlit.  Git
"""






st.write('')
st.write('')


st.subheader('ワインの品質とダイヤモンドの価格予想してみた！')
st.markdown('**sample_data**.')

data_1 = st.checkbox('ワインの品質data')
if data_1:
    wine_data = pd.read_csv('分類data/winequality-red.csv',encoding='shift_jis')
    wine_data = wine_data.drop(wine_data.columns[0],axis=1)
    st.dataframe(wine_data)

data_2 = st.checkbox('ワインの品質purpose_data')
if data_2:
    wine_test = pd.read_csv('分類data/winequality_test.csv',encoding='shift_jis')
    wine_test = wine_test.drop(wine_test.columns[0],axis=1)
    st.dataframe(wine_test)

data_3 = st.checkbox('ダイヤモンドの価格data')
if data_3:
    dia_data = pd.read_csv('回帰data/data.csv',encoding='shift_jis')
    dia_data = dia_data.drop(dia_data.columns[0],axis=1)
    st.dataframe(dia_data)

data_4 = st.checkbox('ダイヤモンドの価格purpose_data')
if data_4:
    dia_test = pd.read_csv('回帰data/test.csv',encoding='shift_jis')
    dia_test = dia_test.drop(dia_test.columns[0],axis=1)
    st.dataframe(dia_test)
st.write('')
st.write('')
st.write('')


task = st.selectbox('機械学習タスクを選択',('','分類', '回帰'))



if task == '分類':
    st.write('')
    uploaded_files = st.file_uploader("予測したいCSV file(porpse_data)：今回はsampleを使用するのでskip", accept_multiple_files= False)
    
#ファイルがアップロードされたら以下が実行される
    if uploaded_files:
        df = pd.read_csv(uploaded_files)
        df = df.drop(df.columns[0],axis=1)
        st.dataframe(df)
    st.write('')



    task = st.selectbox('model選択(上位3つ)',('','rf_model','et_model','lightgbm_model'))

    if task == 'rf_model':
        final_model = load_model('分類data/tuned_model_bun')

        #new_prediction = predict_model(final_model, data=df)
        dia_test = pd.read_csv('分類data/winequality_test.csv',encoding='shift_jis')

        dia_test = dia_test.drop(dia_test.columns[0],axis=1)

        new_prediction = predict_model(final_model, data=dia_test)

        st.subheader('label:予測結果')
        st.balloons()
        st.table(new_prediction)

    if task == 'et_model':
        """
        設定しとらん
        """
    if task == 'lightgbm_model':
        """
        こっちもない
        """
    st.write('')
    st.write('')
    st.write('')


    
    st.subheader('構築モデルの性能分析')

    uploaded_file = st.file_uploader("training_CSV file(data):今回はsampleを使用するのでskip", accept_multiple_files= False)
    #ファイルがアップロードされたら以下が実行される
    if uploaded_file:
        dt = pd.read_csv(uploaded_file)
        dt = dt.drop(dt.columns[0],axis=1)
        st.dataframe(dt)
    st.write('')
    st.write('')

    if st.button('modelの比較'):
        
        modeling = pd.read_csv('分類data/modeling_bun.csv',encoding='shift_jis')
        
        """
        交差検証後，精度が高い順に表示.(※解説2.5)
        """
        st.write(modeling)
    st.write('')
    st.write('')


    task = st.selectbox('model選択',('','rf_model','et_model','lightgbm_model'))

    if task == 'rf_model':
        st.subheader('Hyperparameter/tuning')
        cross_v = pd.read_csv('分類data/cross-validation_bun.csv',encoding='shift_jis')
        tuning = pd.read_csv('分類data/tuning_bun.csv',encoding='shift_jis')
        
        """
        ・※解説3
        """
        """
        ・ランダムグリッドサーチが採用されていますが、現在の主流はベイズ最適化やステップワイズ法
        """
        """
        ・全体的に精度落ちているような…やらない方がいい？…パラメータ調整の検討
        """
        """
        ・今回はデフォルト設定でチューニング後に,model作成
        """
        """
        チューニング前
        """
        st.write(cross_v)
        """
        後
        """
        st.write(tuning)

        @st.cache
        def cross(cross_v):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return cross_v.to_csv().encode('utf-8-sig')
        

        @st.cache
        def tun(tuning):

            return tuning.to_csv().encode('utf-8-sig')
        
        pre_csv = cross(cross_v)
        st.download_button(
            label="pre_tuning.CSV出力",
            data=pre_csv,
            file_name='cross-validation_bun.csv',
            mime='csv',
        )

        post_csv = tun(tuning)
        st.download_button(
            label="post_tuning.CSV出力",
            data=post_csv,
            file_name='tuning_bun.csv',
            mime='csv',
        )







        st.subheader('model性能分析結果と考察')

        if st.button('rf_model分析'):
            """
            ・※解説4.5.6
            """
            image_AUC = Image.open("分類data/AUC.png")
            st.image(image_AUC)


            image_confusion = Image.open("分類data/confusion Matrix.png")
            st.image(image_confusion)


            image_precision = Image.open("分類data/precision recall.png")
            st.image(image_precision)

            image_error = Image.open("分類data/prediction_error_bun.png")
            st.image(image_error)

            image_class = Image.open("分類data/class report.png")
            st.image(image_class)

            image_learning = Image.open("分類data/learning curve_bun.png")
            st.image(image_learning)

            image_validation_b = Image.open("分類data/validation curve_bun.png")
            st.image(image_validation_b)

            image_dementions = Image.open("分類data/dementions.png")
            st.image(image_dementions)

            image_importance2 = Image.open("分類data/feature importance2.png")
            st.image(image_importance2)

            image_decision = Image.open("分類data/decision boundary.png")
            st.image(image_decision)










if task == '回帰':
    st.write('')
    uploaded_files = st.file_uploader("予測したいCSV file(porpse_data)：今回はsampleを使用するのでskip", accept_multiple_files= False)
#ファイルがアップロードされたら以下が実行される
    if uploaded_files:
        df = pd.read_csv(uploaded_files)
        df = df.drop(df.columns[0],axis=1)
        st.dataframe(df)
    st.write('')
    


    task = st.selectbox('model選択(上位3つ)',('','et','rf','lightgbm'))


    if task == 'et':
        final_model = load_model('回帰data/tuned_model1')

        #new_prediction = predict_model(final_model, data=df)
        dia_test = pd.read_csv('回帰data/test.csv',encoding='shift_jis')
        dia_test = dia_test.drop(dia_test.columns[0],axis=1)
        new_prediction = predict_model(final_model, data=dia_test)

        st.subheader('label:予測結果')
        st.balloons()
        st.table(new_prediction)

    if task == 'rf':
        """
        設定しとらん
        """

    if task == 'lightgbm':
    
        """
        こっちもない
        """
    st.write('')
    st.write('')
    st.write('')

    st.subheader('構築モデルの性能分析')

    uploaded_file = st.file_uploader("training_CSV file(data):今回はsampleを使用するのでskip", accept_multiple_files= False)
    #ファイルがアップロードされたら以下が実行される
    if uploaded_file:
        dt = pd.read_csv(uploaded_file)
        dt = dt.drop(dt.columns[0],axis=1)
        st.dataframe(dt)
    st.write('')
    st.write('')    

    if st.button('modelの比較'):
        
        modeling = pd.read_csv('回帰data/modeling.csv',encoding='shift_jis')
        
        """
        交差検証後，精度が高い順に表示.※(解説2.7)
        """
        st.write(modeling)
    st.write('')
    st.write('')


    task = st.selectbox('model選択',('','et_model','rf_model','lightgbm_model'))

    if task == 'et_model':
        st.subheader('Hyperparameter/tuning')
        cross_v = pd.read_csv('回帰data/cross-validation.csv',encoding='shift_jis')
        tuning = pd.read_csv('回帰data/tuning.csv',encoding='shift_jis')
        st.markdown("[※解説4: ハイパーパラメータと最適化](https://www.codexa.net/hyperparameter-tuning-python/)")
        
        """
        ・※解説3
        """
        """
        ・ランダムグリッドサーチが採用されていますが、現在の主流はベイズ最適化やステップワイズ法
        """
        """
        ・R2下がっているような…やらない方がいい？…パラメータ調整の検討
        """
        """
        ・今回はデフォルト設定でチューニング後に,model作成
        """
        """
        チューニング前
        """
        st.write(cross_v)
        """
        後
        """
        st.write(tuning)






        st.subheader('model性能分析結果と考察')
        if st.button('et_model分析'):
            """
            ・※解説4.7.8
            """
            image_Residuals = Image.open("回帰data/Residuals.png")
            st.image(image_Residuals)


            image_Prediction = Image.open("回帰data/Prediction Error.png")
            st.image(image_Prediction)


            image_Cooks = Image.open("回帰data/Cooks Distance.png")
            st.image(image_Cooks)


            image_Lerning = Image.open("回帰data/Lerning Curve.png")
            st.image(image_Lerning)


            image_Validation = Image.open("回帰data/Validation Curve.png")
            st.image(image_Validation)


            image_Manifold = Image.open("回帰data/Manifold Learning.png")
            st.image(image_Manifold)


            image_feature = Image.open("回帰data/feature importance plot.png")
            st.image(image_feature)

        






