import joblib
import pandas as pd
import streamlit as st
from io import BytesIO


# 모델 로드
save_file = 'rc_trained_0219_divided.pkl'
save_data = joblib.load(save_file)

# 로드된 데이터 변수
label_encoder = save_data['label_encoder']
models = save_data['models']
selected_features = save_data['selected_features']
scores = save_data['scores']
mapes = save_data['mapes']
multi_index = save_data['multi_index']
scaler = save_data['scaler']  # 단일 스케일러 로드

# 신규 수량 예측 함수
def predict(target, new_data):
    if target in models:
        model = models[target]
        model_name = model.__class__.__name__
        features = selected_features[target]
        feature_simple = [i[0] for i in features]
        score = scores[target]
        mape = mapes[target]
        # 전체 new_data를 스케일링
        new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns, index=new_data.index)
        # 스케일링된 데이터에서 필요한 피처만 선택
        x_scaled = new_data_scaled[list(features)]
        predict_y = model.predict(x_scaled)

        if predict_y < 0 :
            predict_y = 0
            score = 'N/A'
            mape = 'N/A'
            feature_simple = 'N/A'
            return predict_y, score, mape, feature_simple, model_name
        elif score <= 0 :
            score = 'None'
            mape = 'None'
            return predict_y[0], score, mape, feature_simple, model_name
        else :
            return predict_y[0], score, mape, feature_simple, model_name
    else:
        print('타겟데이터 에러')
        return None


# Streamlit 앱 타이틀
st.title("철근 콘크리트 신속발주 실행 산출")

# 현장명 입력 받기
site_name = st.text_input("현장명을 입력하세요")


# 폼을 사용하여 사용자 입력 받기
with st.form(key='input_form'):
    input_data = []
    errors = []

    # 입력 칸을 두 개의 열로 분할
    columns = st.columns(2)
# # 사용자 입력 받기
# input_data = []
# errors = []

    # 사용자 입력을 위한 Streamlit 위젯
    for idx, feature in enumerate(multi_index):
        with columns[idx % 2]:
            if idx == 9:
                value = st.text_input(f"{feature[0]}").strip().lower()
                value = value.replace('공법','')
                if value in ['0', 'deck', 'de','d']:
                    value = 0
                elif value in ['1', 'pc', 'p']:
                    value = 1
                else:
                    errors.append(f"{feature[0]}을 제대로 입력하세요.")
                    value = None
            else:
                # "0.0"을 표시하지 않기 위해 숫자를 위한 text input 사용
                value = st.text_input(f"{feature[0]}")
                # value = input(f"{feature[0]} 입력: ")
                try:
                    value = float(value)  # 가능하면 float로 변환
                    if idx not in [3, 4, 9] and value <= 0:
                        errors.append(f'{feature[0]}는 0보다 커야 합니다.')
                except ValueError:
                    errors.append(f'{feature[0]}에는 숫자를 입력해야 합니다.')
                    value = None

            input_data.append(value)

    submit_button = st.form_submit_button(label='입력 완료')

if submit_button :

# 입력 검증
    if errors:
        st.error("입력 오류:")
        print("입력 오류:")
        for error in errors:
            st.error(f"- {error}")
            print(f"- {error}")
    else:
        # 데이터프레임 생성
        new_data = pd.DataFrame([input_data], columns=multi_index)

        # 예측 수행
        predictions = {}

        for target in models.keys():
            if (input_data[3] == 0 and target[0] == '상가') or (input_data[4] == 0 and target[0] == '부속동'):
                predictions[target] = {
                    '수량': '추정 안 함',
                    '모델': 'N/A',
                    '신뢰도': 'N/A',
                    '정확도': 'N/A',
                    '특성': 'N/A'
                }
            else:
                predict_y, score, mape, feature_simple, model_name = predict(target, new_data)
                predictions[target] = {
                    '수량': predict_y,
                    '모델': model_name,
                    '신뢰도': score,
                    '정확도': mape,
                    '특성': feature_simple
                }

        new_prediction = pd.DataFrame(predictions)

        # 도메인 지식 적용
        building_types = ['아파트', '부속동', '주차장', '상가']
        indexs = [0, 4]

        for i in indexs:
            for building in building_types:
                try:
                    new_prediction[(building, '철근가공', '현장')].iloc[i] = new_prediction[(building, '철근조립', '현장,스페이서및세퍼레이터포함')].iloc[i] / 33
                except:
                    pass
                try:
                    new_prediction[(building, '철근가공', '현장,DECK')].iloc[i] = new_prediction[(building, '철근조립', '현장,DECK')].iloc[i] / 33
                except:
                    pass
                try:
                    new_prediction[(building, '시스템비계설치', '발판,계단포함,골조')].iloc[i] = new_prediction[(building, '시스템비계해체', '발판,계단포함,골조')].iloc[i]
                except:
                    pass
                try:
                    new_prediction[(building, '시스템비계사용료')].iloc[i] = new_prediction[(building, '시스템비계해체')].iloc[i]
                except:
                    pass
                try:
                    new_prediction[(building, 'SYSTEM SUPPORT손료')].iloc[i] = new_prediction[(building, 'SYSTEM SUPPORT공임')].iloc[i]
                except:
                    pass
                try:
                    new_prediction[(building, '재래식거푸집손료', '아파트/부속동/상가,Tie제거,폼타이포함')].iloc[i] = new_prediction[(building, '재래식거푸집공임', '아파트/부속동/상가,핀제거포함')].iloc[i]
                except:
                    pass
                try:
                    new_prediction[(building, '거푸집정리비', '100%(부속동)')].iloc[i] = new_prediction[(building, '재래식거푸집공임')].iloc[i]
                except:
                    pass
            try:
                new_prediction[('아파트', 'AL FORM 해체')].iloc[i] = new_prediction[('아파트', 'AL FORM 설치')].iloc[i]
            except:
                pass
            try:
                new_prediction[('아파트', '거푸집정리비', '100%(갱폼제외)')].iloc[i] = new_prediction[('아파트', 'AL FORM 설치')].iloc[i]
            except:
                pass
            try:
                new_prediction[('아파트', '거푸집손료', 'AL FORM,부자재')].iloc[i] = new_prediction[('아파트', 'AL FORM 설치')].iloc[i]
            except:
                pass
            try:
                new_prediction[('아파트', '거푸집손료', '갱폼,부자재')].iloc[i] = new_prediction[('아파트', 'GANG FORM 공임')].iloc[i]
            except:
                pass
            try:
                new_prediction[('아파트', '복합단열재(설치)')].iloc[i] = new_prediction[('복합단열재(자재)')].iloc[i]
            except:
                pass
            try:
                new_prediction[('주차장', '주차장거푸집공임')].iloc[i] = new_prediction[('주차장', '거푸집정리비', '100%(주차장)')].iloc[i]
            except:
                pass
            try:
                new_prediction[('주차장', '주차장거푸집손료')].iloc[i] = new_prediction[('주차장', '거푸집정리비', '100%(주차장)')].iloc[i]
            except:
                pass
            try:
                new_prediction[('주차장', '잭써포트사용료')].iloc[i] = new_prediction[('주차장', '잭써포트설치해체')].iloc[i] * 0.7
            except:
                pass
            try:
                new_prediction[('주차장', '와이어메쉬')].iloc[i] = new_prediction[('주차장', '와이어메쉬깔기')].iloc[i]
            except:
                pass

            if input_data[9] == 1:  # PC
                try:
                    new_prediction[('주차장', '철써포트/데크보하부')].iloc[i] = 0
                except:
                    pass
                try:
                    new_prediction[('주차장', '철써포트/데크하부')].iloc[i] = 0
                except:
                    pass
                try:
                    new_prediction[('주차장', '철근가공', '현장,DECK')].iloc[i] = 0
                except:
                    pass
                try:
                    new_prediction[('주차장', '철근가공', '현장,보하부')].iloc[i] = 0
                except:
                    pass
                try:
                    new_prediction[('주차장', '철근조립', '현장,DECK')].iloc[i] = 0
                except:
                    pass
                try:
                    new_prediction[('주차장', '철근조립', '현장,보하부')].iloc[i] = 0
                except:
                    pass

        # Unnamed spec.
        new_columns = [
            (col[0], col[1], '', col[3]) if 'Unnamed' in col[2] else col
            for col in new_prediction.columns
        ]
        new_prediction.columns = pd.MultiIndex.from_tuples(new_columns)


        # 전치
        t_new_prediction = new_prediction.T.reset_index()
        t_new_prediction.columns = ['부위', '명칭', '규격', '단위', '수량', '머신러닝 알고리즘', 'R2 Score', '1-MAPE Score', '예측에 사용한 현장 특성']
        pd.set_option('display.max_row', None)

        t_new_prediction['예측에 사용한 현장 특성'] = t_new_prediction['예측에 사용한 현장 특성'].apply(lambda x: [str(x)] if not isinstance(x, list) else x)

        # 시각적으로 결과 표시
        st.subheader("예측 결과")
        # st.text(t_new_prediction)
        st.dataframe(t_new_prediction)

        print (t_new_prediction)

    if st.form :
        buffer = BytesIO()
        t_new_prediction.to_excel(buffer, index=False)
        buffer.seek(0) #버퍼의 첫포인트로 이동
        st.download_button(
            label="다운로드",
            data=buffer,
            file_name=f"{site_name}_RC_prediction.xlsx",
            mime="application/vnd.ms-excel"
        )
