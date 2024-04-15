import streamlit as st
from streamlit_option_menu import option_menu
import tempfile  # tempfile 모듈을 임포트합니다.
import os
from PIL import Image, ImageDraw
import io  # io 모듈을 임포트합니다.
# import cv2
import numpy as np
from pdf2image import convert_from_path
from pdf2image import convert_from_bytes
from io import BytesIO
import fitz
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
import plotly.figure_factory as ff
import pandas as pd
from datetime import datetime, timedelta
from streamlit_calendar import calendar

# 경고를 무시하도록 설정
warnings.filterwarnings('ignore', message='Conversion of an array with ndim > 0 to a scalar is deprecated')
# # 나중에 모든 경고를 다시 활성화하려면
# warnings.filterwarnings('default', message='Conversion of an array with ndim > 0 to a scalar is deprecated')




time = np.arange(1, 300)
b_slab = 1000
height_tendon = 60
P_tension = 195.8*(45*1000/11500)
# print(P_tension)
DL = 1
LL = 0.1

def Concrete_proterty_analysis():

    fc = np.zeros(time.size+1)

    # KCI 2007
    if Data_36 == 1:
        if Data_37 == 1:  # 습윤양생
            if Data_38 == 1:  # 1종
                fc[1:] = fc_use * np.exp(0.35 * (1 - (28 / time)**0.5))
            elif Data_38 == 2:  # 2종
                fc[1:] = fc_use * np.exp(0.40 * (1 - (28 / time)**0.5))
            else:  # 3종
                fc[1:] = fc_use * np.exp(0.25 * (1 - (28 / time)**0.5))
        else:  # 증기양생
            if Data_38 == 1:  # 1종
                fc[1:] = fc_use * np.exp(0.15 * (1 - (28 / time)**0.5))
            elif Data_38 == 2:  # 2종
                fc[1:] = fc_use * np.exp(0.40 * (1 - (28 / time)**0.5))
            else:  # 3종
                fc[1:] = fc_use * np.exp(0.12 * (1 - (28 / time)**0.5))

    # ACI 209
    elif Data_36 == 2:
        if Data_37 == 1:  # 습윤양생
            if Data_38 == 1:  # 1종
                fc[1:] = fc_use * time / (4 + 0.85 * time)
            else:  # 3종
                fc[1:] = fc_use * time / (2.3 + 0.92 * time)
        else:  # 증기양생
            if Data_38 == 1:  # 1종
                fc[1:] = fc_use * time / (1 + 0.95 * time)
            else:  # 3종
                fc[1:] = fc_use * time / (0.7 + 0.98 * time)
    
    # CEB-FIP 1990
    elif Data_36 == 3:
        if Data_38 == 1:  # 1종
            fc[1:] = fc_use * np.exp(0.25 * (1 - (28 / time)**0.5))
        elif Data_38 == 2:  # 2종
            fc[1:] = fc_use * np.exp(0.38 * (1 - (28 / time)**0.5))
        else:  # 3종
            fc[1:] = fc_use * np.exp(0.20 * (1 - (28 / time)**0.5))
    
    # # Modified Arrehnius (interval of 6 hours)
    # else:
    #     Ta = Data_39 #Temp1
    #     Tb = Data_40 #Temp2
    #     Tc = Data_41 #Temp3

    #     t_trans1=int(24/6*Data_42) # 24/6*Data_42 # day 1 of changing curing temp.
    #     t_trans2=int(24/6*Data_43) # day 2 of changing curing temp.

    #     time_6h = np.arange(0.25, time.max() , 0.25)
    #     time_temp = np.full((4 * time.max(),), Tc)
    #     time_temp[0:t_trans1] = Ta
    #     time_temp[t_trans1:t_trans2] = Tb

    #     Energy = np.zeros(max(time) * 4)  # Energy 배열을 0으로 초기화

    #     for i in range(max(time) * 4):
    #         if time_temp[i] < 20:
    #             Energy[i] = 33.5 + 1.47 * (20 - time_temp[i])
    #         else:
    #             Energy[i] = 33.5

    #     R = 8.3144  # J/(mol·K), 보편적 가스 상수
    #     # Energy 배열에 기반한 감마 계산
    #     gamma = np.exp(Energy * 1000 / R * (1 / (273 + 20) - 1 / (273 + time_temp)))

    #     # 수정된 Arrhenius 식을 위한 상수 계산
    #     modified_const_a = 0.7 + 0.015 * time_temp
    #     modified_const_b = 1.15 - 0.0075 * time_temp

    #     # 수정된 감마 계산
    #     modi_gamma = (1 + time_6h ** 2) / (modified_const_a + time_6h ** 2)
    #     modi_Su = np.sum((1 + time_6h[:4*28] ** 2) / (modified_const_b[:4*28] + time_6h[:4*28] ** 2)) / (4 * 28)

    #     gamma *= modi_gamma

    #     # 수정된 시간에 대한 계산
    #     t_e = np.zeros_like(gamma)
    #     t_e[0] = 0.25 * gamma[0]
    #     for i in range(1, len(gamma)):
    #         t_e[i] = t_e[i-1] + 0.25 * gamma[i]

    #     # 최종 강도 계산
    #     Su = 1 / modi_Su * fc_use
    #     k_r = 0.3  # 일별 비율 상수
    #     fc_6h = Su * (k_r * t_e) / (1 + k_r * t_e)

    #     # app.time에 해당하는 강도 선택
    #     fc = np.zeros_like(time)
    #     for i in range(len(time)):
    #         fc[i] = fc_6h[4 * i]  # 6시간 간격의 데이터를 사용


    if Data_44 == 1:  # KCI 2007 & CEB-FIP 1990
        Ec[1:] = 8500 * fc ** (1/3)
    elif Data_44 == 2:  # ACI 209
        Ec = 4700 * fc ** (1/2)
    else:  # Proposed
        Ec = 4100 * fc ** (1/2)


    # 처음 28일에 대한 Ec 그래프 데이터
    Ec_graph = Ec[:28]
    day_graph = np.arange(1, 29)  # 1부터 28까지의 날짜 배열 생성

    # fcm 계산
    fcm = 1.2 * fc

    #      # 그래프 생성
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=time, y=fc, mode='lines+markers', name='fc'))

    #     # 제목과 레이블 추가
    #     fig.update_layout(
    #         title='콘크리트 강도 발현 곡선',
    #         xaxis_title='시간 (일)',
    #         yaxis_title='콘크리트 강도 (fc) [MPa]'
    # )

    #     # Streamlit 웹사이트에 그래프 출력
    #     st.plotly_chart(fig)     




    return fc, Ec

def construction_load_analyis_before():

    #-------------------------------------#
    L = Span * k_eff
    n_shore = round(L/s1 -1)
    story = top_story - n_slab + 1
    Ig = (s2*h**3)/12

    if partial_shoring_removal:
        n_shore_Al = round(L/s3 - 1)
        Ig_Al = (s4*h**3)/12
    #-------------------------------------#

    #---- Elastic Modulus ----#
    fc, Ec = Concrete_proterty_analysis()

    # Print_fc_graph()

    fc_story = np.zeros([top_story+1,top_story*50])
    a = 0
    for i in range(0,top_story*50):
        fc_story[0,i] = i
    for i in range(1,top_story+1):
        for j in range(a,a+fc.size):
            fc_story[i,j] = fc[j-a]
            if j == top_story*50:
                break
        a += cycle
    
    # st.write(fc_story)

    # Print_Ec_graph()

    Ec_story = np.zeros([top_story+1,top_story*50])
    a = 0
    for i in range(0,top_story*50):
        Ec_story[0,i] = i
    for i in range(1,top_story+1):
        for j in range(a,a+fc.size):
            Ec_story[i,j] = Ec[j-a]
            if j == top_story*50:
                break
        a += cycle

    # st.write(Ec_story)
    #-------------------------#

    #-------------------------#
    LR_cri = cal_LR_cri(fc,L) 

    C = 0.382*10**10*h**2/L**4 # coeff. in crack effect equation

    k = Esh * Ash / s1 / height # spring coeff converted by shore
    # st.write('k',k)

    if partial_shoring_removal:
        k_Al = Esh_Al * Ash_Al / s3 / height # spring coeff converted by Al_support
        # st.write('k_Al',k_Al)
    #-------------------------#
        
    #--- initial load distribution 1~n_slab ---#
    SL = np.zeros([top_story+1,fc_story.size])

    # st.write(top_story)
    day_cast = np.zeros(top_story)
    for i in range(1,top_story):
        day_cast[i] = (i-1) * cycle
    # st.write('day_cast',day_cast)
    day_remove = np.zeros(top_story)
    for i in range(1,top_story):
        day_remove[i] = day_cast[i+n_slab] + remove
        if i+n_slab == top_story-1:
            break
    # st.write('day_remove',day_remove)
    day_remove_Al = np.zeros(top_story)
    for i in range(1,top_story):
        day_remove_Al[i] = day_cast[i+1] + remove_Al
        if i+1 == top_story-1:
            break

    for i in range(0,fc_story.size):
        SL[0,i] = i     

    initial_day = 2 

    while initial_day <= n_slab:
        # st.write(initial_day)
        if initial_day == 2:
            Load_slab = 2 * DL
        else:
            Load_slab = DL
    
        # sum_E = 0
        for j in range(1,initial_day):
            sum_E = 0
            for a in range(1,initial_day):
                # st.write(k)
                # st.write(castday[j])
                sum_E +=  Ec_story[a,int(day_cast[initial_day])]
            # st.write('sum_E',sum_E)
            # st.write('E',Ec_story[j,int(castday[initial_day-1])])

            # for k in range(1,initial_day):
            load_dist = Load_slab * Ec_story[j,int(day_cast[initial_day])]/sum_E
            # st.write('load_dist',load_dist)

            # st.write('i',j,'j:',int(castday[initial_day-1]))
            SL[j,int(day_cast[initial_day]):] += load_dist



        initial_day += 1
    #-------------------------------------#
        
    #--- consider crack effect ---#
    Ie_ratio = np.zeros([top_story+1,SL.size])
    for i in range(0,SL.size):
        Ie_ratio[0,i] = i
    Ie_ratio[1:,:] = 1  
    # st.write('Ie_ratio',Ie_ratio[1:,:100])
    #-------------------------------------#
            
    #--- beta ---#
    beta = np.zeros([top_story+1,SL.size])
    for i in range(0,SL.size):
        beta[0,i] = i
    beta_Al = np.zeros([top_story+1,SL.size])
    for i in range(0,SL.size):
        beta_Al[0,i] = i
    #------------#
            
    
        
    # SL, Ie_ratio, beta, last_day, check = Casting_slab_new(SL, n_slab+1, LR_cri, day_cast, int(day_cast[n_slab+1]), 0, Ie_ratio, C, fc_story, beta, k, Ec_story, Ig, n_shore, L, s1, 1)

    # SL, Ie_ratio, beta, last_day, check = Remove_shore_new(SL, 1, LR_cri, day_remove, int(day_remove[1]), last_day, Ie_ratio, C, fc_story, beta, k, Ec_story, Ig, n_shore, L, s1, check)

    for i in range(1,11):
        last_day = 0
        check = 100 # 상관없는 아무 값으로 시작
        SL, Ie_ratio, beta, last_day, check = Casting_slab_new(SL, n_slab+i, LR_cri, day_cast, int(day_cast[n_slab+i]), last_day, Ie_ratio, C, fc_story, beta, k, Ec_story, Ig, n_shore, L, s1, check)
        SL, Ie_ratio, beta, last_day, check = Remove_shore_new(SL, i, LR_cri, day_cast, int(day_remove[i]), last_day, Ie_ratio, C, fc_story, beta, k, Ec_story, Ig, n_shore, L, s1, check)



    for i in range(11,top_story-n_slab):
        SL, Ie_ratio, beta, last_day, check = Casting_slab_new(SL, n_slab+i, LR_cri, day_cast, int(day_cast[n_slab+i]), last_day, Ie_ratio, C, fc_story, beta, k, Ec_story, Ig, n_shore, L, s1, check)

        if partial_shoring_removal:
            SL, Ie_ratio, beta, beta_Al, last_day, check = Remove_Al_new(SL, n_slab+i-1, LR_cri, day_remove, int(day_remove_Al[n_slab+i-1]), last_day, Ie_ratio, C, fc_story, beta, beta_Al, k, k_Al, Ec_story, Ig, Ig_Al, n_shore, n_shore_Al, L, s1, s2, check)

    
        SL, Ie_ratio, beta, last_day, check = Remove_shore_new(SL, i, LR_cri, day_cast, int(day_remove[i]), last_day, Ie_ratio, C, fc_story, beta, k, Ec_story, Ig, n_shore, L, s1, check)
    

 

    # st.write('SL',SL[1:35,0:400])

   

    return SL, day_cast, day_remove


def Casting_slab_new(SL, cast_floor, LR_cri, day_cast, today, last_day, Ie_ratio, C, fc_story, beta, k, Ec_story, Ig, n_shore, L, s1, check):
    
    #--- Casting slab ---#
    # print('\n',cast_floor,'층 바닥 슬래브 타설')
    # st.info(f"{cast_floor}층 바닥 슬래브 타설"...)
    #--------------------#

    #--- consider crack effect ---#
    if cast_floor > n_slab+1:
        for i in range(1,n_slab+1):
            if Ie_ratio[cast_floor-i,last_day] == 1:
                if SL[cast_floor-i,today] > LR_cri[today - int(day_cast[cast_floor-i])]:
                    Ie_ratio[cast_floor-i,today:] = 1 / (4 - C * fc_story[cast_floor-i,today]/(SL[cast_floor-i,today])**2)
                else:
                    Ie_ratio[cast_floor-i,today:] = 1
    #-----------------------------#
                    
    #--- Load distribution ---#
    for i in range(1,n_slab+1):
        beta[cast_floor-i,today] =( k / (4 * Ec_story[cast_floor-i,today] * (Ig * Ie_ratio[cast_floor-i,today])))**0.25
        # st.write('print_E',Ec_story[cast_floor-i,today])
        # st.write('k',k)
        # st.write('print_Ie_ratio',Ie_ratio[cast_floor-i,today])
        # st.write('beta',beta[cast_floor-i,today])

    y = np.zeros([n_slab,n_shore])
    for t in range(2,n_slab+2):
        for i in range(1,n_shore+1):
            beta_value = beta[cast_floor-(t-1),today]
            # print(j+n_slab, Cast_day[1,j+n_slab+t-1])
            y[t-2,i-1] = (1/k) * (1 - 2 * np.sin(beta_value * L / 2 ) * np.sinh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * np.sin(beta_value * (s1 * i - L/2)) * np.sinh(beta_value * (s1 * i - L/2))\
                                    - 2 * np.cos(beta_value * L / 2) * np.cosh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * (np.cos(beta_value * (s1 * i - L/2)) * np.cosh(beta_value * (s1 * i - L/2))))

            # sin_hx = np.sinh(beta_value * (s1 * i - L/2))
            # print('sin_hx = ',sin_hx)
            # x = beta_value * (s1 * i - L/2)
            # print('x = ',x)

            # print('y = ',y[t-2,i-1])
            # print('beta_value = ',beta_value)
            # asdf = beta_value * L / 2
            # print('asdf = ',asdf)
            # sin_asdf = np.sin(asdf)
            # print('sin_asdf = ',sin_asdf)
            # sin_h = np.sinh(beta_value * L / 2)
            # print('sin_h = ',sin_h)
    # print(y)  
    # st.write('y',y)   

    if y.shape[1] == 1:
        y_sum = y
    else:
        # NumPy의 sum 함수를 사용하여 y의 각 행의 합을 계산합니다.
        # axis=1 파라미터는 행에 대한 합을 계산하라는 것을 의미합니다.
        # keepdims=True 파라미터는 결과를 열 벡터 형태로 유지하도록 합니다.
        y_sum = np.sum(y, axis=1, keepdims=True)        
    # print(y_sum)                                                                                                     
    # st.write('y_sum',y_sum)

    K = ((L/1000)-(y_sum*k*s1/1000))/(y_sum*k*s1/1000) 
    # st.write('K',K)
    # print('\nK = \n',K)       

    EI = np.zeros([n_slab,1])
    for i in range(1,n_slab+1):
        EI[i-1,0] = Ec_story[cast_floor-i,today] * Ie_ratio[cast_floor-i,today]
    # st.write('EI',EI)
    # print('\nEI = \n',EI)    

    A = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A[i,i] += EI[i]
    # print('\nA = \n',A)

    for i in range(0,n_slab-1):
        A[i,i] += EI[i]/K[i]
        A[i+1,i] += -EI[i]/K[i]
        A[i,i+1] += -EI[i]/K[i]
        A[i+1,i+1] += EI[i]/K[i]
    # print('\nA = \n',A)       
    # st.write('A',A)

    Load = np.zeros(n_slab)
    Load[0] = DL + LL
    def_py = np.linalg.inv(A) @ Load.T
    # print('\ndef = \n',def_py)
    # st.write('def_py',def_py*10000)

    A = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A[i,i] += EI[i]
    # print('\nA = \n',A)

    Load_dist = A @ def_py
    # print('\nLoad_dist = ',Load_dist)
    # st.write('Load_dist',Load_dist)

    for i in range(0,n_slab):
        # st.write(SL[cast_floor-i-1,today],'+',Load_dist[i],'=' )
        SL[cast_floor-i-1,today:] += Load_dist[i] 
        # st.write(SL[cast_floor-i-1,today])
    
    check = 0 # 0: cast, 1: remove, 2: remove_al, 3: post_tension

    return SL, Ie_ratio, beta, today, check
    
def Remove_shore_new(SL, remove_floor, LR_cri, day_cast, today, last_day, Ie_ratio, C, fc_story, beta, k, Ec_story, Ig, n_shore, L, s1, check):

    #--- Remove shores ---#
    # print('\n',remove_floor ,'층 동바리 제거')
    # st.info(f"{remove_floor}층 동바리 제거")

    #--- consider crack effect ---#
    for i in range(1,n_slab+1):
        if Ie_ratio[remove_floor+n_slab-i, today] == 1:
            if SL[remove_floor+n_slab-i, today] > LR_cri[today - int(day_cast[remove_floor+n_slab-i])]:
                Ie_ratio[remove_floor+n_slab-i, today:] = 1 / (4 - C * fc_story[remove_floor+n_slab-i, today] / (SL[remove_floor+n_slab-i, today])**2)
        else:
            Ie_ratio[remove_floor+n_slab-i, today:] = min(Ie_ratio[remove_floor+n_slab-i, today], 1 / (4 - C * fc_story[remove_floor+n_slab-i, today] / (SL[remove_floor+n_slab-i, today])**2))
    #-----------------------------#
            
    #--- Load distribution ---#
    for i in range(1,n_slab):
        beta[remove_floor+n_slab-i,today] = ( k/ (4 * Ec_story[remove_floor+n_slab-i, today] * (Ig * Ie_ratio[remove_floor+n_slab-i,today])))**0.25
        
    y = np.zeros([n_slab,n_shore])
    # print('\n')
    for t in range(1,n_slab+1):
        for i in range(1,n_shore+1):
            beta_value = beta[remove_floor+n_slab-(t-1),today]
            # print(j+n_slab , Remove_day[1,j+n_slab+t-1])
            # print('beta_value = ', beta_value)

            y[t-1,i-1] = (1/k) * (1 - 2 * np.sin(beta_value * L / 2 ) * np.sinh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * np.sin(beta_value * (s1 * i - L/2)) * np.sinh(beta_value * (s1 * i - L/2))\
                                - 2 * np.cos(beta_value * L / 2) * np.cosh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * (np.cos(beta_value * (s1 * i - L/2)) * np.cosh(beta_value * (s1 * i - L/2))))
        # print('y = ',y[t-1,i-1])
    # print(y)    
    # st.write('y',y)

    if y.shape[1] == 1:
        y_sum = y
    else:
        # NumPy의 sum 함수를 사용하여 y의 각 행의 합을 계산합니다.
        # axis=1 파라미터는 행에 대한 합을 계산하라는 것을 의미합니다.
        # keepdims=True 파라미터는 결과를 열 벡터 형태로 유지하도록 합니다.
        y_sum = np.sum(y, axis=1, keepdims=True)        
    # print(y_sum)   
    # st.write('y_sum',y_sum)

    K_inv = np.zeros(n_slab)
    for i in range(0,n_slab):
        K_inv[n_slab-i-1] = ((L/1000)-(y_sum[i]*k*s1/1000))/(y_sum[i]*k*s1/1000) 
    # print(K_inv)

    EI = np.zeros([n_slab,1])
    for i in range(1,n_slab):
        EI[i-1,0] = Ec_story[remove_floor+n_slab-i,today] * Ie_ratio[remove_floor+n_slab-i,today]
    # print('\nEI = \n',EI)
    EI[n_slab-1,0] = Ec_story[remove_floor,today]
    # print('\nEI = \n',EI)

    A_inv = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A_inv[i,i] += EI[i]
    # print('\nA = \n',A_inv)

    for i in range(0,n_slab-1):
        A_inv[i,i] += EI[i]/K_inv[i]
        A_inv[i+1,i] += -EI[i]/K_inv[i]
        A_inv[i,i+1] += -EI[i]/K_inv[i]
        A_inv[i+1,i+1] += EI[i]/K_inv[i]
    # print('\nA_inv = \n',A_inv)

    Load_inv = np.zeros(n_slab)
    if check == 0:
        Load_inv[0] = SL[remove_floor,today] - DL - LL
    else:
        Load_inv[0] = SL[remove_floor,today] - DL

    # print('Load_inv = \n',Load_inv)
    def_inv_py = np.linalg.inv(A_inv) @ Load_inv.T
    # print('\ndef_inv = \n',def_inv_py)

    A_inv = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A_inv[i,i] += EI[i]
    Load_dist_inv = A_inv @ def_inv_py
    # print('\nLoad_inv_dist = ',Load_dist_inv)
    # st.write('Load_dist_inv',Load_dist_inv)

    # st.write('today',today)

    for i in range(0,n_slab):
        SL[remove_floor+n_slab-i,today:] += Load_dist_inv[n_slab-i-1] 
    SL[remove_floor,today:] = DL 




    check = 1 # 0: cast, 1: remove, 2: remove_al, 3: post_tension

    return SL, Ie_ratio, beta, today, check

def Remove_Al_new(SL, remove_Al_floor, LR_cri, day_cast, today, last_day, Ie_ratio, C, fc_story, beta, beta_Al, k, k_Al, Ec_story, Ig, Ig_Al, n_shore, n_shore_Al, L, s1, s2, check):

    #--- Remove Al supports ---#
    # print('\n',remove_Al_floor,'층 알서포트 제거')
    # st.info(f"{remove_Al_floor}층 알서포트 제거")
    #--------------------#

    # STEP 1

    #--- consider crack effect ---#
    for i in range(1,n_slab+1):
        if Ie_ratio[remove_Al_floor+1-i,last_day] == 1:
            if SL[remove_Al_floor+1-i,last_day] > LR_cri[last_day - int(day_cast[remove_Al_floor+1])]:
                Ie_ratio[remove_Al_floor+1-i,last_day:] = 1 / (4 - C * fc_story[remove_Al_floor+1-i,last_day]/(SL[remove_Al_floor+1-i,last_day])**2)
            else:
                Ie_ratio[remove_Al_floor+1-i,last_day:] = 1
    #-----------------------------#

    #--- Load distribution ---#
    for i in range(1,n_slab+1):
        beta[remove_Al_floor+1-i,last_day] =( k / (4 * Ec_story[remove_Al_floor+1-i,last_day] * (Ig * Ie_ratio[remove_Al_floor+1-i,last_day])))**0.25
        # st.write('print_E',Ec_story[cast_floor-i,today])
        # st.write('k',k)
        # st.write('print_Ie_ratio',Ie_ratio[cast_floor-i,today])
        # st.write('beta',beta[cast_floor-i,today])

    y = np.zeros([n_slab,n_shore])
    for t in range(2,n_slab+2):
        for i in range(1,n_shore+1):
            beta_value = beta[remove_Al_floor+1-(t-1),last_day]
            # print(j+n_slab, Cast_day[1,j+n_slab+t-1])
            y[t-2,i-1] = (1/k) * (1 - 2 * np.sin(beta_value * L / 2 ) * np.sinh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * np.sin(beta_value * (s1 * i - L/2)) * np.sinh(beta_value * (s1 * i - L/2))\
                                    - 2 * np.cos(beta_value * L / 2) * np.cosh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * (np.cos(beta_value * (s1 * i - L/2)) * np.cosh(beta_value * (s1 * i - L/2))))

            # sin_hx = np.sinh(beta_value * (s1 * i - L/2))
            # print('sin_hx = ',sin_hx)
            # x = beta_value * (s1 * i - L/2)
            # print('x = ',x)

            # print('y = ',y[t-2,i-1])
            # print('beta_value = ',beta_value)
            # asdf = beta_value * L / 2
            # print('asdf = ',asdf)
            # sin_asdf = np.sin(asdf)
            # print('sin_asdf = ',sin_asdf)
            # sin_h = np.sinh(beta_value * L / 2)
            # print('sin_h = ',sin_h)
    # print(y)  
    # st.write('y',y)   

    if y.shape[1] == 1:
        y_sum = y
    else:
        # NumPy의 sum 함수를 사용하여 y의 각 행의 합을 계산합니다.
        # axis=1 파라미터는 행에 대한 합을 계산하라는 것을 의미합니다.
        # keepdims=True 파라미터는 결과를 열 벡터 형태로 유지하도록 합니다.
        y_sum = np.sum(y, axis=1, keepdims=True)        
    # print(y_sum)                                                                                                     
    # st.write('y_sum',y_sum)
        
    for i in range(1,n_slab+1):
        beta_Al[remove_Al_floor+1-i,last_day] =( k_Al / (4 * Ec_story[remove_Al_floor+1-i,last_day] * (Ig * Ie_ratio[remove_Al_floor+1-i,last_day])))**0.25

    y_Al = np.zeros([n_slab,n_shore_Al])
    for t in range(2,n_slab+2):
        for i in range(1,n_shore_Al+1):
            beta_value = beta[remove_Al_floor+1-(t-1),last_day]
            y_Al[t-2,i-1] = (1/k) * (1 - 2 * np.sin(beta_value * L / 2 ) * np.sinh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * np.sin(beta_value * (s1 * i - L/2)) * np.sinh(beta_value * (s1 * i - L/2))\
                                    - 2 * np.cos(beta_value * L / 2) * np.cosh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * (np.cos(beta_value * (s1 * i - L/2)) * np.cosh(beta_value * (s1 * i - L/2))))

    if y_Al.shape[1] == 1:
        y_Al_sum = y_Al
    else:
        y_Al_sum = np.sum(y_Al, axis=1, keepdims=True)    
        # st.write('y_Al_sum',y_Al_sum)         
        
    

    K = ((L/1000)-(y_sum*k*s1/1000))/(y_sum*k*s1/1000) 
    # st.write('K',K)
    # print('\nK = \n',K)       
    K_shore = (k_Al*y_Al_sum) / (k*y_sum + k_Al*y_Al_sum)
    # st.write('k_Al*y_Al_sum',k_Al*y_Al_sum)
    # st.write('k*y_sum + k_Al*y_Al_sum',k*y_sum + k_Al*y_Al_sum)
    # st.write(K_shore)

    EI = np.zeros([n_slab,1])
    for i in range(1,n_slab+1):
        EI[i-1,0] = Ec_story[remove_Al_floor+1-i,last_day] * Ie_ratio[remove_Al_floor+1-i,last_day]
    # st.write('EI',EI)
    # print('\nEI = \n',EI)    

    A = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A[i,i] += EI[i]
    # print('\nA = \n',A)

    for i in range(0,n_slab-1):
        A[i,i] += EI[i]/K[i]
        A[i+1,i] += -EI[i]/K[i]
        A[i,i+1] += -EI[i]/K[i]
        A[i+1,i+1] += EI[i]/K[i]
    # print('\nA = \n',A)       
    # st.write('A',A)

    Load = np.zeros(n_slab)
    if check == 0:
        Load[0] = - DL * K_shore[0] - LL
    else:
        Load[0] = - DL * K_shore[0]        
    def_py = np.linalg.inv(A) @ Load.T
    # print('\ndef = \n',def_py)
    # st.write('def_py',def_py*10000)

    A = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A[i,i] += EI[i]
    # print('\nA = \n',A)

    Load_dist = A @ def_py
    # print('\nLoad_dist = ',Load_dist)
    # st.write('Load_dist',Load_dist)

    for i in range(0,n_slab):
        # st.write(SL[cast_floor-i-1,today],'+',Load_dist[i],'=' )
        SL[remove_Al_floor+1-i-1,today:] += Load_dist[i] 
        # st.write(SL[cast_floor-i-1,today])    
    #---------------------------------------------------------#
        
    # STEP 2
    #--- consider crack effect ---#
    for i in range(1,n_slab+1):
        if Ie_ratio[remove_Al_floor+1-i,today] == 1:
            if SL[remove_Al_floor+1-i,today] > LR_cri[today - int(day_cast[remove_Al_floor+1])]:
                Ie_ratio[remove_Al_floor+1-i,today:] = 1 / (4 - C * fc_story[remove_Al_floor+1-i,today]/(SL[remove_Al_floor+1-i,today])**2)
            else:
                Ie_ratio[remove_Al_floor+1-i,today:] = 1
    #-----------------------------#
    
    #--- Load distribution ---#
    for i in range(0,n_slab+1):
        beta[remove_Al_floor+1-i,today] =( k / (4 * Ec_story[remove_Al_floor+1-i,today] * (Ig * Ie_ratio[remove_Al_floor+1-i,today])))**0.25
        # st.write('print_E',Ec_story[cast_floor-i,today])
        # st.write('k',k)
        # st.write('print_Ie_ratio',Ie_ratio[cast_floor-i,today])
        # st.write('beta',beta[cast_floor-i,today])

    y = np.zeros([n_slab+1,n_shore])
    # st.write(n_shore)
    for t in range(1,n_slab+2):
        for i in range(1,n_shore+1):
            beta_value = beta[remove_Al_floor+1-(t-1),today]
            # print(j+n_slab, Cast_day[1,j+n_slab+t-1])
            y[t-1,i-1] = (1/k) * (1 - 2 * np.sin(beta_value * L / 2 ) * np.sinh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * np.sin(beta_value * (s1 * i - L/2)) * np.sinh(beta_value * (s1 * i - L/2))\
                                    - 2 * np.cos(beta_value * L / 2) * np.cosh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * (np.cos(beta_value * (s1 * i - L/2)) * np.cosh(beta_value * (s1 * i - L/2))))
           
            # sin_hx = np.sinh(beta_value * (s1 * i - L/2))
            # print('sin_hx = ',sin_hx)
            # x = beta_value * (s1 * i - L/2)
            # print('x = ',x)

            # print('y = ',y[t-2,i-1])
            # print('beta_value = ',beta_value)
            # asdf = beta_value * L / 2
            # print('asdf = ',asdf)
            # sin_asdf = np.sin(asdf)
            # print('sin_asdf = ',sin_asdf)
            # sin_h = np.sinh(beta_value * L / 2)
            # print('sin_h = ',sin_h)
    # print(y)  
    # st.write('y',y)   

    if y.shape[1] == 1:
        y_sum = y
    else:
        # NumPy의 sum 함수를 사용하여 y의 각 행의 합을 계산합니다.
        # axis=1 파라미터는 행에 대한 합을 계산하라는 것을 의미합니다.
        # keepdims=True 파라미터는 결과를 열 벡터 형태로 유지하도록 합니다.
        y_sum = np.sum(y, axis=1, keepdims=True)  
    # st.write('y_sum',y_sum)

    K = ((L/1000)-(y_sum*k*s1/1000))/(y_sum*k*s1/1000) 
    # st.write('K',K)
    # print('\nK = \n',K)         

    EI = np.zeros([n_slab+1,1])
    for i in range(0,n_slab+1):
        EI[i,0] = Ec_story[remove_Al_floor+1-i,today] * Ie_ratio[remove_Al_floor+1-i,today]
    # st.write('EI',EI)
    # print('\nEI = \n',EI)    

    A = np.zeros([n_slab+1,n_slab+1])
    for i in range(0,n_slab+1):
        A[i,i] += EI[i]
    # print('\nA = \n',A)
    # st.write('A',A)

    for i in range(0,n_slab):
        A[i,i] += EI[i]/K[i]
        A[i+1,i] += -EI[i]/K[i]
        A[i,i+1] += -EI[i]/K[i]
        A[i+1,i+1] += EI[i]/K[i]
    # print('\nA = \n',A)       
        # st.write('A',A)

    Load = np.zeros(n_slab+1)
    Load[0] = DL * K_shore[0]
    def_py = np.linalg.inv(A) @ Load.T
    # print('\ndef = \n',def_py)
    # st.write('def_py',def_py*10000)

    A = np.zeros([n_slab+1,n_slab+1])
    for i in range(0,n_slab+1):
        A[i,i] += EI[i]
    # print('\nA = \n',A)
    # st.write('A',A)

    Load_dist = A @ def_py
    # print('\nLoad_dist = ',Load_dist)
    # st.write('Load_dist',Load_dist)

    for i in range(0,n_slab+1):
        # st.write(SL[cast_floor-i-1,today],'+',Load_dist[i],'=' )
        SL[remove_Al_floor+1-i,today:] += Load_dist[i] 
        # st.write(SL[cast_floor-i-1,today])    
        
    #---------------------------------------------------------#
        
    check = 3 # 0: cast, 1: remove, 2: remove_al, 3: post_tension

    return SL, Ie_ratio, beta, beta_Al, today, check




def cal_LR_cri(f,L):
    
    if post_tensioning: #포스트텐션 적용시

        f_r = (0.63*f**0.5)

        y_2 = h/2 # 콘크리트 단면 중심을 도심으로 가정

        Z_2 = (b_slab*h**3/12)/y_2

        r = (((b_slab*h**3)/12)/(h*b_slab))**0.5
        # print('r = ',r)

        e_p = abs(height_tendon - y_2)

        P_e = 0.85*P_tension
        # print(P_e)

        M_cr_pt = (f_r * Z_2)/1000 + P_e*((r**2)/y_2 + e_p)
        # print('((r**2)/y_2 + e_p) = ',((r**2)/y_2 + e_p))
        # print('(f_r * Z_2)/1000 = ',(f_r * Z_2)/1000)
        # print('P_e*((r**2)/y_2 + e_p) = ',P_e*((r**2)/y_2 + e_p))
        # print(M_cr_pt[1])

        M_cr = f_r * Z_2/1000
        # print(M_cr[1])

        pt_factor = M_cr_pt/M_cr
        # print(pt_factor)

        LR_cri = pt_factor * 0.357 * 10**5 * h / L**2 * f**0.5 # critical load
        # print(LR_cri)
    else:
        LR_cri = 0.357 *10**5 *h / L**2 * f**0.5
        

    # print('LR_cri = ',LR_cri)
    
    # time_LR_cri = np.arange(0,f.max())

    # fig_LR_cri_ = go.Figure()
    # fig_LR_cri_.add_trace(go.Scatter(x=time_LR_cri, y=LR_cri, mode='lines+markers', name='LR_cri'))
    # fig_LR_cri_.update_layout(
    #     title='LR_cri',
    #     xaxis_title='시간 (일)',
    #     yaxis_title='슬래브 하중 (DL)'
    # )
    # st.plotly_chart(fig_LR_cri_)  # Streamlit 웹사이트에 그래프 출력

    return LR_cri

def Print_fc_graph():

    fc, Ec = Concrete_proterty_analysis()
    
#     # 그래프 생성
#     fig_fc = go.Figure()
#     fig_fc.add_trace(go.Scatter(x=time, y=fc, mode='lines+markers', name='fc'))

#     # 제목과 레이블 추가
#     fig_fc.update_layout(
#         title='콘크리트 강도 발현 곡선',
#         xaxis_title='시간 (일)',
#         yaxis_title='콘크리트 강도 (fc) [MPa]'
# )

#     # Streamlit 웹사이트에 그래프 출력
#     st.plotly_chart(fig_fc)

#     # 그래프 생성
#     fig_Ec = go.Figure()kl
#     fig_Ec.add_trace(go.Scatter(x=time, y=Ec, mode='lines+markers', name='Ec'))

#     # 제목과 레이블 추가
#     fig_Ec.update_layout(
#         title='탄성계수 곡선',
#         xaxis_title='시간 (일)',
#         yaxis_title='탄성계수 (Ec) [MPa]'
# )

#     # Streamlit 웹사이트에 그래프 출력
#     st.plotly_chart(fig_Ec)
    
    # 31개의 데이터 포인트만 포함하도록 슬라이싱
    time_sliced = np.zeros(31)
    time_sliced[1:31] = time[:30]
    fc_sliced = fc[:31]
    Ec_s3liced = Ec[:31]
    
    # 콘크리트 강도 발현 곡선 그래프 생성
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=time_sliced, y=fc_sliced, mode='lines+markers', name='fc'))
    fig_fc.update_layout(
        title='콘크리트 강도 발현 곡선',
        xaxis_title='시간 (일)',
        yaxis_title='콘크리트 강도 (fc) [MPa]'
    )
    st.plotly_chart(fig_fc)  # Streamlit 웹사이트에 그래프 출력

    # # 탄성계수 곡선 그래프 생성
    # fig_Ec = go.Figure()
    # fig_Ec.add_trace(go.Scatter(x=time_sliced, y=Ec_sliced, mode='lines+markers', name='Ec'))
    # fig_Ec.update_layout(
    #     title='탄성계수 곡선',
    #     xaxis_title='시간 (일)',
    #     yaxis_title='탄성계수 (Ec) [MPa]'
    # )
    # st.plotly_chart(fig_Ec)  # Streamlit 웹사이트에 그래프 출력

    return

def Print_Ec_graph():

    fc, Ec = Concrete_proterty_analysis()

    time_sliced = np.zeros(31)
    time_sliced[1:31] = time[:30]
    Ec_sliced = Ec[:31]

    # 탄성계수 곡선 그래프 생성
    fig_Ec = go.Figure()
    fig_Ec.add_trace(go.Scatter(x=time_sliced, y=Ec_sliced, mode='lines+markers', name='Ec'))
    fig_Ec.update_layout(
        title='탄성계수 곡선',
        xaxis_title='시간 (일)',
        yaxis_title='탄성계수 (Ec) [MPa]'
    )
    st.plotly_chart(fig_Ec)  # Streamlit 웹사이트에 그래프 출력

    return

def Casting_slab(j, n_slab, Cast_day, Remove_day, Ie_ratio, beta, LR_cri, k, E, Ig, n_shore, L, C, f, SL):

    #--- Casting slab ---#
    print('\n',j + n_slab,'층 바닥 슬래브 타설')
    #--- consider crack effect ---#
    for i in range(1,n_slab+1):
        if SL[j+n_slab-i,Remove_day[1,j+n_slab-1+i]] > LR_cri[Remove_day[1,j+n_slab-1+i]-Cast_day[1,j+n_slab-1]]:
            # if post_tensioning:
            #     Ie_ratio[j+n_slab,Remove_day[1,j+n_slab-1+i]] = 1
            # else:
                Ie_ratio[j+n_slab,Remove_day[1,j+n_slab-1+i]] = 1 / (4 - C * f[Remove_day[1,j+n_slab-1+i]-Cast_day[1,j+n_slab-1+i]]) / (SL[j+n_slab-i, Remove_day[1,j+n_slab-1+i]])**2
    #-----------------------------#

    #--- Load distribution ---#
    for i in range(1,n_slab+1):
        beta[j+n_slab,Cast_day[1,j+n_slab+i]] = (k / 4 / E[Cast_day[1,j+n_slab+i]-Cast_day[1,j+n_slab]] / (Ig * Ie_ratio[j+n_slab+i,Remove_day[1,j+n_slab-1+i]]))**0.25
        # print(E[Cast_day[1,j+n_slab+i]-Cast_day[1,j+n_slab]])
        # print('Ie = ',Ie_ratio[j+n_slab+i,Remove_day[1,j+n_slab-1+i]])
        # print('beta_1 = ',beta[j+n_slab,Cast_day[1,j+n_slab+i]])
    # print('beta = ',beta)
    
    y = np.zeros([n_slab,n_shore])
    for t in range(2,n_slab+2):
        for i in range(1,n_shore+1):
            beta_value = beta[j+n_slab,Cast_day[1,j+n_slab+t-1]]
            # print(j+n_slab, Cast_day[1,j+n_slab+t-1])
            y[t-2,i-1] = (1/k) * (1 - 2 * np.sin(beta_value * L / 2 ) * np.sinh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * np.sin(beta_value * (s1 * i - L/2)) * np.sinh(beta_value * (s1 * i - L/2))\
                                    - 2 * np.cos(beta_value * L / 2) * np.cosh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * (np.cos(beta_value * (s1 * i - L/2)) * np.cosh(beta_value * (s1 * i - L/2))))

            # sin_hx = np.sinh(beta_value * (s1 * i - L/2))
            # print('sin_hx = ',sin_hx)
            # x = beta_value * (s1 * i - L/2)
            # print('x = ',x)

            # print('y = ',y[t-2,i-1])
            # print('beta_value = ',beta_value)
            # asdf = beta_value * L / 2
            # print('asdf = ',asdf)
            # sin_asdf = np.sin(asdf)
            # print('sin_asdf = ',sin_asdf)
            # sin_h = np.sinh(beta_value * L / 2)
            # print('sin_h = ',sin_h)
    # print(y)      

    if y.shape[1] == 1:
        y_sum = y
    else:
        # NumPy의 sum 함수를 사용하여 y의 각 행의 합을 계산합니다.
        # axis=1 파라미터는 행에 대한 합을 계산하라는 것을 의미합니다.
        # keepdims=True 파라미터는 결과를 열 벡터 형태로 유지하도록 합니다.
        y_sum = np.sum(y, axis=1, keepdims=True)        
    # print(y_sum)                                                                                                     
            
    K = ((L/1000)-(y_sum*k*s1/1000))/(y_sum*k*s1/1000) 
    # print('\nK = \n',K)

    EI = np.zeros([n_slab,1])
    for i in range(1,n_slab+1):
        EI[i-1,0] = E[Cast_day[1,j+n_slab+1+i]-Cast_day[1,j+n_slab+1]] * Ie_ratio[j+n_slab+i,Remove_day[1,j+n_slab-1+i]]
    # print('\nEI = \n',EI)

    A = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A[i,i] += EI[i]
    # print('\nA = \n',A)

    for i in range(0,n_slab-1):
        A[i,i] += EI[i]/K[i]
        A[i+1,i] += -EI[i]/K[i]
        A[i,i+1] += -EI[i]/K[i]
        A[i+1,i+1] += EI[i]/K[i]
    # print('\nA = \n',A)

    Load = np.zeros(n_slab)
    Load[0] = DL + LL
    def_py = np.linalg.inv(A) @ Load.T
    # print('\ndef = \n',def_py)
    
    A = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A[i,i] += EI[i]
    # print('\nA = \n',A)

    Load_dist = A @ def_py
    # print('\nLoad_dist = ',Load_dist)

    if j == 14:
        if n_slab == 4:
            for i in range(0,n_slab):
                SL[j+n_slab-1-i,Cast_day[1,j+n_slab]:] += Load_dist[i]
        else:
            for i in range(0,n_slab):
                SL[j+n_slab-1-i,Cast_day[1,j+n_slab]:] += Load_dist[i] 
    else:
        for i in range(0,n_slab):
            SL[j+n_slab-1-i,Cast_day[1,j+n_slab]:] += Load_dist[i] 

    return SL , beta

def Remove_shore(j, n_slab, Cast_day, Remove_day, Ie_ratio, beta, LR_cri, k, E, Ig, n_shore, L, C, f, SL):
    #--- Remove shores ---#
    print('\n',j ,'층 동바리 제거')
    #--- consider crack effect ---#
    for i in range(1,n_slab+1):
        if SL[j+n_slab-i,Cast_day[1,j+n_slab+i+1]] > LR_cri[Cast_day[1,j+n_slab+i+1]-Cast_day[1,j+n_slab+i]]:
            # if post_tensioning:
            #     Ie_ratio[j+n_slab,Cast_day[1,j+n_slab+i+1]] = 1
            # else:
                Ie_ratio[j+n_slab,Cast_day[1,j+n_slab+i+1]] = 1 / (4 - C * f[Cast_day[1,j+n_slab+i+1]-Cast_day[1,j+n_slab+i]] / SL[j+n_slab-i,Cast_day[1,j+n_slab+i+1]])
    #-----------------------------#
                
    #--- Load distribution ---#
    for i in range(1,n_slab):
        beta[j+n_slab,Remove_day[1,j+n_slab+n_slab-i]] = (k / 4 / E[Remove_day[1,j+n_slab+n_slab-i]-Cast_day[1,j+n_slab]] / (Ig * Ie_ratio[j+n_slab+n_slab-i, Cast_day[1,j+n_slab+n_slab-i]]))**0.25
        # print(Remove_day[1,j+n_slab+n_slab-i]-Cast_day[1,j+n_slab])
        # print(E[Remove_day[1,j+n_slab+n_slab-i]-Cast_day[1,j+n_slab+n_slab]])
        # print(j+n_slab, Remove_day[1,j+n_slab+n_slab-i])
        # print('changed beta = ',beta[j+n_slab,Remove_day[1,j+n_slab+n_slab-i]])

    y = np.zeros([n_slab,n_shore])
    # print('\n')
    for t in range(1,n_slab+1):
        for i in range(1,n_shore+1):
            beta_value = beta[j+n_slab,Remove_day[1,j+n_slab+t-1]]
            # print(j+n_slab , Remove_day[1,j+n_slab+t-1])
            # print('beta_value = ', beta_value)

            y[t-1,i-1] = (1/k) * (1 - 2 * np.sin(beta_value * L / 2 ) * np.sinh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * np.sin(beta_value * (s1 * i - L/2)) * np.sinh(beta_value * (s1 * i - L/2))\
                                - 2 * np.cos(beta_value * L / 2) * np.cosh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * (np.cos(beta_value * (s1 * i - L/2)) * np.cosh(beta_value * (s1 * i - L/2))))
        # print('y = ',y[t-1,i-1])
    # print(y)

    if y.shape[1] == 1:
        y_sum = y
    else:
        # NumPy의 sum 함수를 사용하여 y의 각 행의 합을 계산합니다.
        # axis=1 파라미터는 행에 대한 합을 계산하라는 것을 의미합니다.
        # keepdims=True 파라미터는 결과를 열 벡터 형태로 유지하도록 합니다.
        y_sum = np.sum(y, axis=1, keepdims=True)        
    # print(y_sum)   

    K_inv = np.zeros(n_slab)
    for i in range(0,n_slab):
        K_inv[n_slab-i-1] = ((L/1000)-(y_sum[i]*k*s1/1000))/(y_sum[i]*k*s1/1000) 
    # print(K_inv)

    EI = np.zeros([n_slab,1])
    for i in range(1,n_slab):
        EI[i-1,0] = E[Remove_day[1,j+n_slab+n_slab-i]-Cast_day[1,j+n_slab]] * Ie_ratio[j+n_slab+n_slab-i, Cast_day[1,j+n_slab+n_slab-i]]
    # print('\nEI = \n',EI)
    EI[n_slab-1,0] = E[Remove_day[1,j+n_slab]-Cast_day[1,j+n_slab]]
    # print('\nEI = \n',EI)

    A_inv = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A_inv[i,i] += EI[i]
    # print('\nA = \n',A_inv)

    for i in range(0,n_slab-1):
        A_inv[i,i] += EI[i]/K_inv[i]
        A_inv[i+1,i] += -EI[i]/K_inv[i]
        A_inv[i,i+1] += -EI[i]/K_inv[i]
        A_inv[i+1,i+1] += EI[i]/K_inv[i]
    # print('\nA_inv = \n',A_inv)

    Load_inv = np.zeros(n_slab)
    # if post_tensioning:
    if j >= 10:
        Load_inv[0] = SL[j,Remove_day[1,j+n_slab+1]] - DL
    elif n_slab == 4:
        Load_inv[0] = SL[j,Remove_day[1,j+n_slab+1]] - DL - LL
    else:
        Load_inv[0] = SL[j,Remove_day[1,j+n_slab+1]] - DL - LL

    # print('Load_inv = \n',Load_inv)
    def_inv_py = np.linalg.inv(A_inv) @ Load_inv.T
    # print('\ndef_inv = \n',def_inv_py)

    A_inv = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A_inv[i,i] += EI[i]
    Load_dist_inv = A_inv @ def_inv_py
    # print('\nLoad_inv_dist = ',Load_dist_inv)

    if j == 14:
        if n_slab == 4:
            for i in range(0,n_slab):
                SL[j+n_slab-i,Remove_day[1,j+n_slab-1]:] += Load_dist_inv[n_slab-i-1] 
                print(j+n_slab-i)
    else:
        for i in range(0,n_slab):
            SL[j+n_slab-i,Remove_day[1,j+n_slab]:] += Load_dist_inv[n_slab-i-1] 
            print(j+n_slab-i)

    
    SL[j,Remove_day[1,j+n_slab]:] = 1

    return SL, beta

def PT_slab(j, n_slab, Cast_day, Remove_day, Post_tension_day, Ie_ratio, beta, LR_cri, k, E, Ig, n_shore, L, C, f, SL):
    j = j - 1
    #--- Casting slab ---#
    print('\n',j + n_slab,'층 바닥 슬래브 인장')
    #--- consider crack effect ---#
    for i in range(1,n_slab+1):
        if SL[j+n_slab-i,Remove_day[1,j+n_slab-1+i]] > LR_cri[Remove_day[1,j+n_slab-1+i]-Cast_day[1,j+n_slab-1]]:
                Ie_ratio[j+n_slab,Remove_day[1,j+n_slab-1+i]] = 1 / (4 - C * f[Remove_day[1,j+n_slab-1+i]-Cast_day[1,j+n_slab-1+i]]) / (SL[j+n_slab-i, Remove_day[1,j+n_slab-1+i]])**2
    #-----------------------------#
                
    #--- Load distribution ---#
    for i in range(1,n_slab+1):
        beta[j+n_slab,Cast_day[1,j+n_slab+i]] = (k / 4 / E[Cast_day[1,j+n_slab+i]-Cast_day[1,j+n_slab]] / (Ig * Ie_ratio[j+n_slab+i,Remove_day[1,j+n_slab-1+i]]))**0.25
        # print(E[Cast_day[1,j+n_slab+i]-Cast_day[1,j+n_slab]])
        # print('Ie = ',Ie_ratio[j+n_slab+i,Remove_day[1,j+n_slab-1+i]])
        # print('beta_1 = ',beta[j+n_slab,Cast_day[1,j+n_slab+i]])
    # print('beta = ',beta)
    
    y = np.zeros([n_slab,n_shore])
    for t in range(2,n_slab+2):
        for i in range(1,n_shore+1):
            beta_value = beta[j+n_slab,Cast_day[1,j+n_slab+t-1]]
            # print(j+n_slab, Cast_day[1,j+n_slab+t-1])
            y[t-2,i-1] = (1/k) * (1 - 2 * np.sin(beta_value * L / 2 ) * np.sinh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * np.sin(beta_value * (s1 * i - L/2)) * np.sinh(beta_value * (s1 * i - L/2))\
                                    - 2 * np.cos(beta_value * L / 2) * np.cosh(beta_value * L / 2) / (np.cos(beta_value * L) + np.cosh(beta_value * L)) * (np.cos(beta_value * (s1 * i - L/2)) * np.cosh(beta_value * (s1 * i - L/2))))

            # sin_hx = np.sinh(beta_value * (s1 * i - L/2))
            # print('sin_hx = ',sin_hx)
            # x = beta_value * (s1 * i - L/2)
            # print('x = ',x)

            # print('y = ',y[t-2,i-1])
            # print('beta_value = ',beta_value)
            # asdf = beta_value * L / 2
            # print('asdf = ',asdf)
            # sin_asdf = np.sin(asdf)
            # print('sin_asdf = ',sin_asdf)
            # sin_h = np.sinh(beta_value * L / 2)
            # print('sin_h = ',sin_h)
    # print(y)      

    if y.shape[1] == 1:
        y_sum = y
    else:
        # NumPy의 sum 함수를 사용하여 y의 각 행의 합을 계산합니다.
        # axis=1 파라미터는 행에 대한 합을 계산하라는 것을 의미합니다.
        # keepdims=True 파라미터는 결과를 열 벡터 형태로 유지하도록 합니다.
        y_sum = np.sum(y, axis=1, keepdims=True)        
    # print(y_sum)                                                                                                     
            
    K = ((L/1000)-(y_sum*k*s1/1000))/(y_sum*k*s1/1000) 
    # print('\nK = \n',K)

    EI = np.zeros([n_slab+1,1])
    for i in range(1,n_slab+2):
        EI[i-1,0] = E[Cast_day[1,j+n_slab+1+i]-Cast_day[1,j+n_slab+1]] * Ie_ratio[j+n_slab+i,Remove_day[1,j+n_slab-1+i]]
    # print('\nEI = \n',EI)
    
    n_slab = n_slab + 1
    # print('n_slab = ',n_slab)

    A = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A[i,i] += EI[i]
    # print('\nA = \n',A)

    for i in range(0,n_slab-1):
        A[i,i] += EI[i]/K[i]
        A[i+1,i] += -EI[i]/K[i]
        A[i,i+1] += -EI[i]/K[i]
        A[i+1,i+1] += EI[i]/K[i]
    # print('\nA = \n',A)

    Load = np.zeros(n_slab)
    Load[1] = -DL*post_tension_factor - LL
    # print(post_tension_factor)
    def_py = np.linalg.inv(A) @ Load.T
    # print('\ndef = \n',def_py)
    
    A = np.zeros([n_slab,n_slab])
    for i in range(0,n_slab):
        A[i,i] += EI[i]
    # print('\nA = \n',A)

    Load_dist = A @ def_py
    # print('\nLoad_dist = ',Load_dist)
    Load_dist[1] += DL * post_tension_factor

    n_slab = n_slab -1 
    if j == 13:
        if n_slab == 4:
            print(n_slab)   
            for i in range(0,n_slab+1):
                print(j+n_slab-i+1)
                print(Post_tension_day[1,j+n_slab+1])
                SL[j+n_slab-i+1,Post_tension_day[1,j+n_slab+1]:]+= Load_dist[i]
        else:
            for i in range(0,n_slab+1):
                SL[j+n_slab-i+1,Post_tension_day[1,j+n_slab]:] += Load_dist[i] 
    else:
        for i in range(0,n_slab+1):
            SL[j+n_slab-i+1,Post_tension_day[1,j+n_slab]:] += Load_dist[i] 
            print(Post_tension_day[1,j+n_slab])

    j = j + 1

    return SL, beta

                




st.set_page_config(page_title='구조 안전성 평가',\
                   page_icon='c:/Users/user/Desktop/PythonWorkspace/streamlit/images/konkuk_logo.jpg',\
                     layout='wide')


with st.sidebar:
    selected = option_menu("구조 안전성 평가", 
                           ["시공 전 안전성 평가", '시공 중/후 안전성 평가', '장기처짐 평가'], 
                            icons=['none', 'none','none'], 
                            menu_icon="none", 
                            default_index=0,
                            # styles={
                            # "container": {"padding": "0", "background-color": "#ffffaa"},
                            # "icon": {"color": "#ff0000", "font-size": "15px"},
                            # "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#ffccff"},
                            # "nav-link-selected": {"background-color": "#ffaaff"},}
                        )
    



if selected == "시공 전 안전성 평가":
    st.markdown("""
    <h1 style='text-align: center; margin-top: -70px; margin-bottom: 20px;'>시공 전 안전성 평가</h1>
    """, unsafe_allow_html=True)
elif selected == "시공 중/후 안전성 평가":
    st.markdown("<h1 style='text-align: left;'>시공 중/후 안전성 평가</h1>", unsafe_allow_html=True)
elif selected == "장기처짐 평가":
    st.markdown("<h1 style='text-align: left;'>장기처짐 평가</h1>", unsafe_allow_html=True)

if selected == "시공 전 안전성 평가":
    # with st.expander("결과 확인", expanded=True):
    #     # st.write('ㅁㅇㄹ')
    #     st.image("https://static.streamlit.io/examples/dice.jpg")
    #     if st.button('결과 확인'):
    #         construction_load_analyis()
    COL_1, COL_2 = st.columns(2)

    with COL_2:
            
        with st.expander("설계 정보 입력", expanded=True):
            # st.write("##  Example site")

            tab1_1, tab1_2, tab1_3, tab1_4, tab1_5, tab1_6 = st.tabs(
                [
                    "대표 슬래브 설정",
                    "건물 및 슬래브 정보",
                    "가설 계획",
                    "재료 정보",
                    "구조 계산 정보",
                    "콘크리트 정보"

                ]
            )

            with tab1_1:
                st.write("대표 슬래브 영역 설정")

                # # 파일 업로드 위젯 생성
                # uploaded_image = st.file_uploader("이미지 파일 업로드", type=["jpg", "png", "jpeg"])

                # # 업로드된 파일이 있다면
                # if uploaded_image is not None:
                #     # PIL 이미지로 변환
                #     pil_image = Image.open(uploaded_image)

                #     # 이미지 크기 설정
                #     width, height = pil_image.size
                #     st.image(pil_image, caption="원본 이미지", use_column_width=True)

                #     # 이미지 자르기 위젯
                #     st.sidebar.header("이미지 자르기")
                #     crop_x = st.sidebar.slider("X 좌표", 0, width, 0)
                #     crop_y = st.sidebar.slider("Y 좌표", 0, height, 0)
                #     crop_width = st.sidebar.slider("가로 크기", 0, width, width)
                #     crop_height = st.sidebar.slider("세로 크기", 0, height, height)
                    
                #     # 이미지 자르기
                #     cropped_image = pil_image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
                #     st.image(cropped_image, caption="자른 이미지", use_column_width=True)

                #     # OpenCV 이미지로 변환
                #     cv_image = np.array(cropped_image)

                #     # 이미지 회전 위젯
                #     st.sidebar.header("이미지 회전")
                #     rotation_angle = st.sidebar.slider("회전 각도", -180, 180, 0)

                #     # 이미지 회전
                #     rotated_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
                #     st.image(rotated_image, caption="회전한 이미지", use_column_width=True)

                #     # 이미지 그리기 위젯
                #     st.sidebar.header("그리기")
                #     draw = st.sidebar.checkbox("그리기 모드")
                    
                #     if draw:
                #         st.sidebar.write("그리기 모드를 활성화하였습니다.")
                #         draw_color = st.sidebar.color_picker("펜 색상 선택", "#000000")
                #         line_width = st.sidebar.slider("펜 굵기", 1, 10, 2)

                #         # OpenCV 이미지에 그리기
                #         draw_image = cv_image.copy()
                #         draw_image = cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR)
                #         cv_draw = ImageDraw.Draw(cropped_image)

                #         # 그리기 동작
                #         draw_points = st.sidebar.text_area("그리기 동작 (예: 100,100;200,200)", "")
                #         if draw_points:
                #             draw_points = draw_points.split(";")
                #             for points in draw_points:
                #                 points = points.split(",")
                #                 if len(points) == 4:
                #                     x1, y1, x2, y2 = int(points[0]), int(points[1]), int(points[2]), int(points[3])
                #                     cv2.line(draw_image, (x1, y1), (x2, y2), tuple(int(draw_color[i:i+2], 16) for i in (1, 3, 5)), line_width)

                #         st.image(draw_image, caption="그린 이미지", use_column_width=True)
                    
                #     else:
                #         st.sidebar.write("그리기 모드를 활성화하려면 체크하세요.")

                # PDF 파일 업로드
                uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

                if uploaded_file is not None:
                    # 업로드된 PDF 파일을 바이트 데이터로 읽기
                    pdf_bytes = uploaded_file.read()

                    # PDF를 이미지로 변환
                    images = convert_from_bytes(pdf_bytes)

                    # 이미지를 화면에 출력
                    for i, image in enumerate(images):
                        st.image(image, caption=f"Page {i + 1}", use_column_width=True)

                    # PDF 파일 정보 출력
                    st.write(f"Total Pages: {len(images)}")
                else:
                    st.write("Upload a PDF file to get started")
                



                # st.image("https://static.streamlit.io/examples/dice.jpg")

            # 나머지 탭도 동일한 패턴으로 처리

            with tab1_2:
                st.write("건물 및 슬래브 정보")

                col1, col2 = st.columns([1, 2])

                with col2:
                    st.write("")

                with col1:
                    # 슬래브 종류 선택 드롭다운 메뉴 추가
                    slab_type_info = st.selectbox(
                        "슬래브 종류 선택",
                        ["1방향 슬래브", "2방향 슬래브"]
                    )
                    
                    # 세션 상태에서 기본값을 불러오기 위해 안전하게 접근합니다.
                    building_info = st.session_state.get('building_info', {})

                    # 각 입력 필드에 대해 기본값을 지정합니다.
                    building_info['slab_area'] = st.number_input("슬래브 넓이 [㎡]", value=building_info.get('slab_area', 53.37))
                    building_info['slab_thickness'] = st.number_input("슬래브 두께 [mm]", value=building_info.get('slab_thickness', 210))
                    building_info['slab_span'] = st.number_input("스팬 [mm]", value=building_info.get('slab_span', 5330))
                
                col_1, col_2 = st.columns([1,2])
                with col_1:
                    building_info['k_eff'] = st.number_input("유효 스팬 계수", value=building_info.get('k_eff', 0.8))
                with col_2:
                    with st.popover("?"):
                        st.image("https://static.streamlit.io/examples/dice.jpg")

                col1, col2 = st.columns([1, 2])
                with col2:
                    st.write("")

                with col1:
                    # 슬래브 종류 선택 드롭다운 메뉴 추가
                    building_info['slab_height'] = st.number_input("층고 [mm]", value=building_info.get('slab_height', 2820))
                    building_info['top_story'] = st.number_input("층수 [층]", value=building_info.get('top_story', 27))


                # 선택된 슬래브 종류에 따라 추가 정보를 요청하거나 계산 수행
                if slab_type_info == "1방향 슬래브":
                    slab_type = 1
                elif slab_type_info == "2방향 슬래브":
                    slab_type = 2


                slab_area = building_info['slab_area']
                h = building_info['slab_thickness']
                Span = building_info['slab_span']
                k_eff = building_info['k_eff']
                height = building_info['slab_height']
                top_story = building_info['top_story']+10

                # 세션 상태를 업데이트하는 방식을 수정합니다.
                st.session_state['building_info'] = building_info
                print('\n건물 및 슬래브 정보 입력값')
                print('\nslab_type = ',slab_type)
                print('slab_area = ',slab_area)
                print('h (slab_thickness) = ',h)
                print('slab_span = ',Span)
                print('k_eff = ',k_eff)
                print('slab_height = ',height)
                print('top_story = ',top_story,'\n')




            with tab1_3:
                st.write("가설 계획")
        
                # 컬럼 너비를 다르게 설정 (예: 첫 번째 컬럼은 두 번째 컬럼보다 두 배 넓게)
                col1, col2, col3 = st.columns([1, 2, 1])  # 여기서 [2, 1]은 첫 번째 컬럼이 두 번째 컬럼보다 너비가 두 배라는 의미


                with col2:
                    st.write("")

                with col3:
                    partial_shoring_removal = st.checkbox("일부 동바리 제거 여부",value=True)
                    post_tensioning = st.checkbox("포스트텐션 여부")
                    with st.popover("More info"):
                        st.image("https://static.streamlit.io/examples/dice.jpg")

                with col1:
                    construction_plan = st.session_state.get('construction_plan', {})
                    construction_plan['n_slab'] = st.number_input("동바리 지지 층수 [층]", value=construction_plan.get('n_slab', 3))
                    construction_plan['cycle'] = st.number_input("층 당 시공주기 [일]", value=construction_plan.get('cycle', 7))
                    if partial_shoring_removal:
                        construction_plan['remove'] = st.number_input("필러서포트 제거 시점 [일]", value=construction_plan.get('remove', 5))
                        construction_plan['remove_Al'] = st.number_input("일부동바리 제거 시점 [일]", value=construction_plan.get('remove_Al', 3))
                    else:
                        construction_plan['remove'] = st.number_input("동바리 제거 시점 [일]", value=construction_plan.get('remove', 6))
                    if post_tensioning:
                        construction_plan['post_tension'] = st.number_input("포스트 텐션 시점 [일]", value=construction_plan.get('post_tension', 30))
                        construction_plan['post_tension_factor'] = st.number_input("포스트 텐션 상향력 [%]", value=construction_plan.get('post_tension_factor', 100))
                        
                n_slab = construction_plan['n_slab']
                cycle = construction_plan['cycle']
                remove = construction_plan['remove']

                # 선택한 옵션에 따라 추가 정보를 요구하거나 계산 수행
                if partial_shoring_removal:
                    remove_Al = construction_plan['remove_Al']
                else:
                    remove_Al = 0

                if post_tensioning :
                    post_tension = construction_plan['post_tension']
                    post_tension_factor = construction_plan['post_tension_factor']/100
                else:
                    post_tension = 0
                    post_tension_factor = 0

                st.session_state['construction_plan'] = construction_plan
                print('\n가설 계획 입력값')
                print('\nn_slab =',n_slab)
                print('cycle =',cycle)
                print('remove =',remove)
                print('remove_Al =',remove_Al)
                print('post_tension =',post_tension)
                print('post_tension_factor =',post_tension_factor,'\n')
                    


            


            with tab1_4:
                st.write("가설 계획")

                col1, col2, col3 = st.columns([2, 2, 0.5]) 

                with col1:

                    rebar_info = st.selectbox(
                            "슬래브 철근 직경 선택",
                            ["d 4", "d 5", "d 6", "d 7", "d 8", \
                            "d 10", "d 13", "d 16", "d 19",\
                                "d 22", "d 25", "d 29", \
                                    "d 32", "d 35", "d 38", \
                                        "d 41", "d 43", "d 51", "d 57"], index=5
                        )

                    material_info = st.session_state.get('material_info', {})
                    material_info['space_ten_rebar'] = st.number_input("인장철근 간격 [mm]", value=material_info.get('space_ten_rebar', 300))
                    material_info['space_com_rebar'] = st.number_input("압축철근 간격 [mm]", value=material_info.get('space_com_rebar', 300))
                    material_info['fy_design'] = st.number_input("설계 철근 항복 강도 [MPa]", value=material_info.get('fy_design', 500))
                    material_info['upper_covering_dep'] = st.number_input("슬래브 상부 피복 두께 [mm]", value=material_info.get('upper_covering_dep', 30))
                    material_info['bottom_covering_dep'] = st.number_input("슬래브 하부 피복 두께 [mm]", value=material_info.get('bottom_covering_dep', 30))

                    if partial_shoring_removal:
                        material_info['Esh'] = st.number_input("필러서포트 탄성계수 [MPa]", value=material_info.get('Esh', 70000))
                        material_info['Ash'] = st.number_input("필러서포트 단면적 [㎟]", value=material_info.get('Ash', 676.2))
                        material_info['Esh_Al'] = st.number_input("제거하는 동바리 탄성계수 [MPa]", value=material_info.get('Esh_Al', 70000))
                        material_info['Ash_Al'] = st.number_input("제거하는 동바리 단면적 [㎟]", value=material_info.get('Ash_Al', 676.2))
                    else:
                        material_info['Esh'] = st.number_input("동바리 탄성계수 [MPa]", value=material_info.get('Esh', 70000))
                        material_info['Ash'] = st.number_input("동바리 단면적 [㎟]", value=material_info.get('Ash', 5757.0))


                with col2:
                    if partial_shoring_removal:
                        material_info['num_sh'] = st.number_input("대표 슬래브를 지지하는 필러서포트 개수 [개]", value=material_info.get('num_sh', 24))
                        material_info['num_sh_Al'] = st.number_input("제거하는 동바리 개수 [개]", value=material_info.get('num_sh_Al', 29))
                    else:
                        material_info['num_sh'] = st.number_input("대표 슬래브를 지지하는 동바리 개수 [개]", value=material_info.get('num_sh', 41))

                if rebar_info == "d 4":
                    a_rebar = 14.05
                    d_rebar_value = 4
                elif rebar_info == "d 5":
                    a_rebar = 21.98
                    d_rebar_value = 5
                elif rebar_info == "d 6":
                    a_rebar = 31.67
                    d_rebar_value = 6
                elif rebar_info == "d 7":
                    a_rebar = 38.48
                    d_rebar_value = 7
                elif rebar_info == "d 8":
                    a_rebar = 49.51
                    d_rebar_value = 8
                elif rebar_info == "d 10":
                    a_rebar = 71.33
                    d_rebar_value = 10
                elif rebar_info == "d 13":
                    a_rebar = 126.7
                    d_rebar_value = 13
                elif rebar_info == "d 16":
                    a_rebar = 21.98
                    d_rebar_value = 16
                elif rebar_info == "d 19":
                    a_rebar = 31.67
                    d_rebar_value = 19
                elif rebar_info == "d 22":
                    a_rebar = 387.1
                    d_rebar_value = 22
                elif rebar_info == "d 25":
                    a_rebar = 506.7
                    d_rebar_value = 25
                elif rebar_info == "d 29":
                    a_rebar = 642.4
                    d_rebar_value = 29
                elif rebar_info == "d 32":
                    a_rebar = 794.2
                    d_rebar_value = 32
                elif rebar_info == "d 35":
                    a_rebar = 956.6
                    d_rebar_value = 35
                elif rebar_info == "d 38":
                    a_rebar = 1140
                    d_rebar_value = 38
                elif rebar_info == "d 41":
                    a_rebar = 1340
                    d_rebar_value = 41
                elif rebar_info == "d 43":
                    a_rebar = 1452
                    d_rebar_value = 43
                elif rebar_info == "d 51":
                    a_rebar = 2027
                    d_rebar_value = 51
                elif rebar_info == "d 57":
                    a_rebar = 2579
                    d_rebar_value = 57

                space_ten_rebar = material_info['space_ten_rebar'] 
                space_com_rebar = material_info['space_com_rebar'] 
                fy_design = material_info['fy_design']
                upper_covering_dep = material_info['upper_covering_dep']
                bottom_covering_dep = material_info['bottom_covering_dep']
                Esh = material_info['Esh']
                Ash = material_info['Ash']
                num_sh = material_info['num_sh']

                if partial_shoring_removal:
                    Esh_Al = material_info['Esh_Al']
                    Ash_Al = material_info['Ash_Al']
                    num_sh_Al = material_info['num_sh_Al']
                else:
                    Esh_Al = 0
                    Ash_Al = 0
                    num_sh_Al = 0

                As = a_rebar*space_ten_rebar
                d_slab = h - upper_covering_dep - (d_rebar_value/2) # h-yt

                s1 = 1000*(slab_area/num_sh)**0.5
                s2 = 1000*(slab_area/num_sh)**0.5
                if partial_shoring_removal:
                    s3 = 1000*(slab_area/num_sh_Al)**0.5
                    s4 = 1000*(slab_area/num_sh_Al)**0.5

                st.session_state['material_info'] = material_info
                print('\n재료 정보 입력값')
                print('a_rebar = ',a_rebar)
                print('d_rebar_value = ',d_rebar_value)
                print('space_ten_rebar = ',space_ten_rebar)
                print('space_com_rebar = ',space_com_rebar)
                print('fy_design = ',fy_design)
                print('upper_covering_dep = ',upper_covering_dep)
                print('bottom_covering_dep = ',bottom_covering_dep)
                print('Esh = ',Esh)
                print('Ash = ',Ash)
                print('num_sh = ',num_sh)
                print('num_sh_Al = ',num_sh_Al)
                print('As = ',As)
                print('d_slab = ',d_slab)
                print('s1 = ',s1)
                if partial_shoring_removal:
                    print('s3 = ',s3)
            




            with tab1_5:
                st.write("구조 계산 정보 입력 (구조계산서 참고)")

                col1, col2, col3 = st.columns([0.1, 1, 1]) 

                with col1:
                    st.latex(r"\large M_u")
                    st.write('\n\n\n')
                    st.latex(r"\large \gamma_D")
                    st.write('\n\n\n')
                    st.latex(r"\large \gamma_L")
                    st.write('\n\n\n')
                    st.latex(r"\large \gamma_C")
                    st.write('\n\n\n')
                    st.write('\n\n\n')
                    st.write('\n\n\n')
                    st.latex(r"\large D")
                    st.write('\n\n\n')
                    st.latex(r"\large L")
                    st.write('\n\n\n')
                    st.latex(r"\large \Phi")
                    st.write('\n\n\n')
                    st.latex(r"\large \Phi_c")


                with col2:
                    structural_analysis_info = st.session_state.get('structural_analysis_info', {})
                    structural_analysis_info['M_u'] = st.number_input("설계 정모멘트 [kN · m/m]", value=structural_analysis_info.get('M_u', 18.45))
                    structural_analysis_info['gamma_D'] = st.number_input("(default = 1.2) ", value=structural_analysis_info.get('gamma_D', 1.0))
                    structural_analysis_info['gamma_L'] = st.number_input("(default = 1.6) ", value=structural_analysis_info.get('gamma_L', 1.6))
                    structural_analysis_info['gamma_C'] = st.number_input("(default = 1.0) ", value=structural_analysis_info.get('gamma_C', 1.0))
                    structural_analysis_info['D'] = st.number_input("(고정하중) [N/㎡] ", value=structural_analysis_info.get('D', 5.04))
                    structural_analysis_info['L'] = st.number_input("(활하중) [N/㎡]", value=structural_analysis_info.get('L', 2.5))
                    structural_analysis_info['phi'] = st.number_input("(휨강도 안전율. 강도감소계수) ", value=structural_analysis_info.get('phi', 0.85))
                    structural_analysis_info['phi_c'] = st.number_input("(시공하중 안전율, default - 1.0) ", value=structural_analysis_info.get('phi_c', 1.0))

                M_u = structural_analysis_info['M_u']
                gamma_D = structural_analysis_info['gamma_D']
                gamma_L = structural_analysis_info['gamma_L'] 
                gamma_C = structural_analysis_info['gamma_C']
                D_value = structural_analysis_info['D']
                L_value = structural_analysis_info['L']
                phi = structural_analysis_info['phi'] 
                phi_c = structural_analysis_info['phi_c']

                st.session_state['structural_analysis_info'] = structural_analysis_info



                



            with tab1_6:
                st.write("콘크리트 정보 입력")

                col1, col2 = st.columns([1, 2])
                with col1:
                    concrete_info = st.session_state.get('concrete_info', {})
                    concrete_info['fc_use'] = st.number_input("설계 강도 [MPa]", value=concrete_info.get('fc_use', 40))
                    
                col1, col22, col33, col44 = st.columns([1, 0.4, 0.8, 0.8]) 
                with col1:
                    concrete_strength_development_info = st.selectbox(
                            "강도 발현식 선택",
                            ["KDS", "ACI 209", "CEB-FIP 1990", "Modified Arrehnius"]
                        )
                    



                col1, col2 = st.columns([1, 2])

                with col1:
                 
                    concrete_curing_method_info = st.selectbox(
                            "양생 방법 선택",
                            ["습윤 양생", "증기 양생"]
                        )
                    cement_type_info = st.selectbox(
                            "시멘트 종류 선택",
                            ["1종 시멘트", "2종 시멘트", "3종 시멘트"]
                        )
                
                col1, col23 = st.columns([1, 2])
                with col1:
                    Elastic_modulus_of_concrete_info = st.selectbox(
                            "콘트리트 탄성 계수식 선택",
                            ["KDS & CEB-FIP 1990", "ACI 209", "보정식"], index=2
                        )
                
                with col33:
                    if concrete_strength_development_info == "Modified Arrehnius":
                        concrete_info['Data_42'] = st.number_input("양생 온도 변화 시점(1) [일]", value=concrete_info.get('Data_42', 5))
                        concrete_info['Data_43'] = st.number_input("양생 온도 변화 시점(2) [일]", value=concrete_info.get('Data_43', 10))
                    
                with col44:
                    if concrete_strength_development_info == "Modified Arrehnius":
                        concrete_info['Data_39'] = st.number_input("양생 온도 변화 시 양생 온도(1) [°C]", value=concrete_info.get('Data_39', 20))
                        concrete_info['Data_40'] = st.number_input("양생 온도 변화 시 양생 온도(2) [°C]", value=concrete_info.get('Data_40', 25))
                        concrete_info['Data_41'] = st.number_input("양생 온도 변화 시 양생 온도(3) [°C]", value=concrete_info.get('Data_41', 20))


                fc_use = concrete_info['fc_use'] + 8

                if concrete_strength_development_info == "KDS":
                    Data_36 = 1
                elif concrete_strength_development_info == "ACI 209":
                    Data_36 = 2
                elif concrete_strength_development_info == "CEB-FIP 1990":
                    Data_36 = 3
                elif concrete_strength_development_info == "Modified Arrehnius":
                    Data_36 = 4

                if concrete_curing_method_info == "습윤 양생":
                    Data_37 = 1
                elif concrete_curing_method_info == "증기 양생":
                    Data_37 = 2

                if cement_type_info == "1종 시멘트":
                    Data_38 = 1
                elif cement_type_info == "2종 시멘트":
                    Data_38 = 2
                elif cement_type_info == "3종 시멘트":
                    Data_38 = 3

                if concrete_strength_development_info == "Modified Arrehnius":
                    Data_42 = concrete_info['Data_42']
                    Data_43 = concrete_info['Data_43']
                    Data_39 = concrete_info['Data_39']
                    Data_40 = concrete_info['Data_40']
                    Data_41 = concrete_info['Data_41']
                else:
                    Data_42 = 0
                    Data_43 = 0
                    Data_39 = 0
                    Data_40 = 0
                    Data_41 = 0

                if Elastic_modulus_of_concrete_info == "KDS & CEB-FIP 1990":
                    Data_44 = 1
                elif Elastic_modulus_of_concrete_info == "ACI 209":
                    Data_44 = 2
                elif Elastic_modulus_of_concrete_info == "보정식":
                    Data_44 = 3

                st.session_state['concrete_info'] = concrete_info

                fc_use = concrete_info['fc_use'] + 8

                with col22:
                    with st.popover("More info"):
                        Print_fc_graph()    

                with col23:
                    with st.popover("More info"):
                        Print_Ec_graph()       



    with COL_1:

        with st.expander("안전성 평가 결과", expanded=True):
            col_1, col_2,col_3 = st.columns([7,1,1])

            with col_1:
                check_live_result = st.checkbox('결과 확인')
            with col_3:
                floors = [f"{i}층" for i in range(1, top_story -9)]
                selected_floor = st.selectbox('층 선택', floors, label_visibility="collapsed")
                
            tab_1, tab_2, tab_3 = st.tabs(
                [
                    "휨강도 안전성 평가",
                    "균열 저항 성능 평가",
                    "전단 강도 안전성 평가",
                ]
                )
            

            with tab_1:
                # col_1, col_2 = st.columns([9, 1])
                
                if check_live_result and selected_floor:
                    check_floor = int(selected_floor.replace('층', ''))
                    SL, day_cast, day_remove = construction_load_analyis_before()

                    # Index calculation
                    start_idx = check_floor + 10
                    end_idx = check_floor + 10
                    # st.write(int(day_cast[start_idx]))
                    # st.write(int(day_remove[end_idx])+1)
                    # st.write(SL[check_floor+10,int(day_cast[start_idx]):int(day_remove[end_idx])+1])

                    # Data preparation
                    x_values = np.arange(0, int(day_remove[end_idx]) - int(day_cast[start_idx]) + 1)
                    data_to_plot = SL[check_floor+10,int(day_cast[start_idx]):int(day_remove[end_idx])+1]

                    # Create the graph
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_values, y=data_to_plot, mode='lines+markers', name='Data'))
                    fig.update_layout(
                        title='휨강도 안전성 평가',
                        xaxis_title='재령(일)',
                        yaxis_title='시공하중(DL)',
                        # template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab_2:
                # col_1, col_2 = st.columns([9, 1])
                
                if check_live_result and selected_floor:
                    check_floor = int(selected_floor.replace('층', ''))
                    SL, day_cast, day_remove = construction_load_analyis_before()

                    # Index calculation
                    start_idx = check_floor + 10
                    end_idx = check_floor + 10
                    # st.write(int(day_cast[start_idx]))
                    # st.write(int(day_remove[end_idx])+1)
                    # st.write(SL[check_floor+10,int(day_cast[start_idx]):int(day_remove[end_idx])+1])

                    # Data preparation
                    x_values = np.arange(0, int(day_remove[end_idx]) - int(day_cast[start_idx]) + 1)
                    data_to_plot = SL[check_floor+10,int(day_cast[start_idx]):int(day_remove[end_idx])+1]

                    # Create the graph
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_values, y=data_to_plot, mode='lines+markers', name='Data'))
                    fig.update_layout(
                        title='균열 저항 성능 평가',
                        xaxis_title='재령(일)',
                        yaxis_title='시공하중(DL)',
                        # template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)


                with tab_3:
                # col_1, col_2 = st.columns([9, 1])
                
                    if check_live_result and selected_floor:
                        check_floor = int(selected_floor.replace('층', ''))
                        SL, day_cast, day_remove = construction_load_analyis_before()

                        # Index calculation
                        start_idx = check_floor + 10
                        end_idx = check_floor + 10
                        # st.write(int(day_cast[start_idx]))
                        # st.write(int(day_remove[end_idx])+1)
                        # st.write(SL[check_floor+10,int(day_cast[start_idx]):int(day_remove[end_idx])+1])

                        # Data preparation
                        x_values = np.arange(0, int(day_remove[end_idx]) - int(day_cast[start_idx]) + 1)
                        data_to_plot = SL[check_floor+10,int(day_cast[start_idx]):int(day_remove[end_idx])+1]

                        # Create the graph
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x_values, y=data_to_plot, mode='lines+markers', name='Data'))
                        fig.update_layout(
                            title='전단강도 안전성 평가',
                            xaxis_title='재령(일)',
                            yaxis_title='시공하중(DL)',
                            # template='plotly_dark'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            
            
 



if selected == "시공 중/후 안전성 평가":
    # with st.expander("결과 확인", expanded=True):
    #     # st.write('ㅁㅇㄹ')
    #     st.image("https://static.streamlit.io/examples/dice.jpg")
    #     if st.button('결과 확인'):
    #         construction_load_analyis()
    COL_1, COL_2 = st.columns(2)

    # CON_1, CON_2 = st.container(2)

    

    with COL_2:   
        tab1_1, tab1_2 = st.tabs(
            [
                "시공 일정 입력",
                "재료 시험"
                

            ]
        )

        with tab1_1:
        
        # event_types = ["타설", "최하단 동바리 제거", "알서포트 제거", "인장"]
        # selected_event_type = st.selectbox("이벤트 유형 선택:", event_types)

        # events = [
        #     {
        #         "title": "10층 타설",
        #         "color": "#FF6C6C",
        #         "start": "2024-04-03",
        #         "end": "2024-04-05",
        #         "resourceId": "a",
        #     }    
        #     ]
        
        # # 이벤트 유형에 따른 색상 변경
        # if selected_event_type == "타설":
        #     events[0]["color"] = "#0000FF"  # 파란색
        # elif selected_event_type == "최하단 동바리 제거":
        #     events[0]["color"] = "#008000"  # 녹색
        # elif selected_event_type == "알서포트 제거":
        #     events[0]["color"] = "#FF4500"  # 오렌지색
        # elif selected_event_type == "인장":
        #     events[0]["color"] = "#FF4500"  # 오렌지색
        # else:
        #     events[0]["color"] = "#0000FF"  # 파란색

            # mode = st.selectbox(
            #     "Calendar Mode:",
            #     ("daygrid", "multimonth"),
            # )
            mode = "daygrid"

            COL_date_1, COL_date_2 = st.columns([8.5,1.3])

            with COL_date_2:

                with st.popover("일정 추가"):

                    # 이벤트 제목과 날짜 입력
                    event_title = st.text_input("이벤트 제목", "")
                    event_start = st.date_input("날짜", datetime.today())
                    # event_end = st.date_input("종료 날짜", datetime.today())



                    # 세션 상태에서 'events'를 확인하고, 없으면 초기화
                    if 'events' not in st.session_state:
                        st.session_state['events'] = []

                    # 고유 키를 초기화하지 않은 경우, 초기화
                    if 'calendar_key' not in st.session_state:
                        st.session_state['calendar_key'] = datetime.now().isoformat()

                    # 이벤트 추가 버튼 클릭 시 동작
                    if st.button("이벤트 추가"):
                        new_event = {
                            "title": event_title,
                            "color": "#FF6C6C",
                            "start": event_start.isoformat(),
                            "end": event_start.isoformat(),
                            "resourceId": "a",
                        }
                        st.session_state['events'].append(new_event)
                        st.session_state['calendar_key'] = datetime.now().isoformat()

            # 캘린더 리소스 및 옵션 설정
            # calendar_resources = [{"id": "a", "building": "Building A", "title": "Room A"}]
            if mode == "daygrid":
                calendar_options = {
                    "editable": True,
                    "selectable": True,
                    "initialView": "dayGridMonth",
                    "headerToolbar": {
                        "left": "today",
                        "center": "title",
                        "right": "prev,next",
                    }  }
            elif mode ==  "multimonth":
                calendar_options = {
                    "editable": True,
                    "selectable": True,
                    "initialView": "multiMonthYear",
                    "headerToolbar": {
                        "left": "today",
                        "center": "title",
                        "right": "prev,next",
                    } 
                }


            # 캘린더 컴포넌트 호출
            calendar_component = calendar(
                events=st.session_state.get("events", []),
                options=calendar_options,
                key=st.session_state['calendar_key']
            )

            if calendar_component.get("eventsSet") is not None:
                # 이벤트 데이터가 제대로된 딕셔너리 형태인지 검사
                if all(isinstance(event, dict) and 'id' in event for event in calendar_component["eventsSet"]):
                    updated_events = {event['id']: event for event in calendar_component["eventsSet"]}
                    st.session_state['events'] = list(updated_events.values())
                else:
                    st.error("반환된 이벤트 데이터가 예상 형식과 다릅니다.")

            # 세션 상태의 이벤트 목록 출력
            if st.session_state['events']:
                st.write("이벤트가 세션 상태에 추가되었습니다:", st.session_state['events'])
            else:
                st.write("아직 이벤트가 추가되지 않았습니다.")

            # 임시로 이벤트 목록 출력
            st.write("세션 상태의 이벤트 목록:", st.session_state.get('events', []))

        with tab1_2:
                st.write("재료 시험")



    with COL_1:
            st.write('ㅁㅇㄹ')
            st.image("https://static.streamlit.io/examples/dice.jpg")








    # st.write(state)

    # st.markdown("## API reference")
    # st.help(calendar)
        

    








# if selected == "시공 전 안전성 평가":
#     col1, col2 = st.columns(2)
#     with col1:
#         with st.expander("그래프 출력", expanded=True):
#             st.write('ㅁㅇㄹ')
#             st.image("https://static.streamlit.io/examples/dice.jpg")

#     with col2:
#         with st.expander("입력", expanded=True):
#             st.write('ㅁㅇㄹ')
#             st.image("https://static.streamlit.io/examples/dice.jpg")

# streamlit run c:/Users/user/Desktop/PythonWorkspace/streamlit/app.py
            


