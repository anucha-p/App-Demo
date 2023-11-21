import streamlit as st
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_color_lut, apply_windowing, apply_voi_lut
import numpy as np
from xml.etree import ElementTree
import pandas as pd
import pyodide_http # https://github.com/whitphx/stlite/blob/main/packages/desktop/README.md
pyodide_http.patch_all()
# import predictor
import streamlit.components.v1 as components

st.set_page_config(page_title="App Title", layout='wide')
st.title("Demo")

st.markdown(f"""
    <style>
        .appview-container .main .block-container{{
            max-width: {1200}px;
            padding-top: {5}rem;
            padding-right: {1}rem;
            padding-left: {1}rem;
            padding-bottom: {10}rem;
        }}

    </style>
    """,
    unsafe_allow_html=True)


# response = requests.request(url="https://streamlit.io/",method='GET')
# st.write(response.status_code)

# model_path = "/mnt/ssd/python_projects/MPI_pred/streamlit_lite/streamlit_app/model.pth"
# model = predictor.load_model(model_path)

def read_dcm(dcm_img):
    ds = dcmread(dcm_img)
    arr = ds.pixel_array
    # rgb = apply_color_lut(arr, palette='PET')
    arr = apply_modality_lut(arr, ds)
    # arr = apply_voi_lut(arr, ds, index=0)
    arr = apply_color_lut(arr, palette='PET')
    # img = ds.pixel_array.astype(float)
    return arr

def normalize(x):
    norm_x = (x-np.min(x))/(np.max(x)-np.min(x))
    return norm_x

def display_polar(ds):
    arr = ds.pixel_array
    arr = normalize(arr)
    # arr = apply_color_lut(arr, palette='PET')
    st.image(arr, use_column_width=True)

def check_id(ds, df):
    ds_id = str(ds.PatientID)
    if df.loc['id'].str.contains(ds_id).any():
        pt_id = ':green[Patient ID: '+ ds_id + ']'
    else:
        pt_id = ':red[Patient ID: ' + ds_id + ']'
    st.write(pt_id)    

def display_xml(xml):
    root = xml.getroot()

    x = root[0].attrib
    df = pd.DataFrame.from_dict(x, orient='index')
    st.dataframe(df, use_container_width=True)


def read_xml_stress(root):

    x = root[0].attrib
    # if x['sex'] == 'Male':
    #     x['sex'] = 1
    # elif x['sex'] == 'Female':
    #     x['sex'] = 0
    # else:
    #     x['sex'] = 'NA'
    
    # if x['wgtUnit'] == 'lbs':
    #     w = x['weight']
    #     w = w * 0.453592
    #     x['weight'] = w
    #     x['wgtUnit'] = 'kg'
    
    # if x['hgtUnit'] == 'in':
    #     h = x['height'] 
    #     h = h * 2.54
    #     x['height'] = h
    #     x['wgtUnit'] = 'cm'
    
    # STRESS
    # Perfusion
    sum_score = root[2][1][1].attrib
    x['per_summedScore'] = sum_score['summedScore']

    per_score = dict()
    for i in range(17):
        region = root[2][1][1][i].attrib
        segment_name = 'per_' + list(region.values())[0]
        per_score[segment_name] = list(region.values())[1]
    x.update(per_score)

    LV_per = dict()
    for i in range(4):
        region = root[2][1][2][i].attrib
        segment_name = 'per_' + list(region.values())[0]
        LV_per[segment_name] = list(region.values())[1]
    x.update(LV_per)

    # Motion
    motion = root[2][2][1].attrib
    x['motion_summedScore'] = motion['summedScore']
    motion_score = dict()
    for i in range(17):
        region = root[2][2][1][i].attrib
        segment_name = 'motion_' + list(region.values())[0]
        motion_score[segment_name] = list(region.values())[1]
    x.update(motion_score)

    # Thickness
    thick = root[2][2][2].attrib
    x['thick_summedScore'] = motion['summedScore']
    thick_score = dict()
    for i in range(17):
        region = root[2][2][2][i].attrib
        segment_name = 'thick_' + list(region.values())[0]
        thick_score[segment_name] = list(region.values())[1]
    x.update(thick_score)

    # LV Function
    lv_fun = root[2][2][3].attrib
    lv_dict = dict()
    for key, value in lv_fun.items():
        new_key = 's_' + key
        lv_dict[new_key] = value
    x.update(lv_dict)

    return x


def read_xml_str_rst(root):
    # STRESS
    x = read_xml_stress(root)
    # xml = ElementTree.parse(xml_file)
    # root = xml.getroot()
    
    # REST
    # Perfusion
    sum_score = root[3][1][1].attrib
    x['s_per_summedScore'] = sum_score['summedScore']

    per_score = dict()
    for i in range(17):
        region = root[3][1][1][i].attrib
        segment_name = 'r_per_' + list(region.values())[0]
        per_score[segment_name] = list(region.values())[1]
    x.update(per_score)

    LV_per = dict()
    for i in range(4):
        region = root[3][1][2][i].attrib
        segment_name = 'r_per_' + list(region.values())[0]
        LV_per[segment_name] = list(region.values())[1]
    x.update(LV_per)

    # Motion
    motion = root[3][2][1].attrib
    x['r_motion_summedScore'] = motion['summedScore']
    motion_score = dict()
    for i in range(17):
        region = root[3][2][1][i].attrib
        segment_name = 'r_motion_' + list(region.values())[0]
        motion_score[segment_name] = list(region.values())[1]
    x.update(motion_score)

    # Thickness
    thick = root[3][2][2].attrib
    x['r_thick_summedScore'] = motion['summedScore']
    thick_score = dict()
    for i in range(17):
        region = root[3][2][2][i].attrib
        segment_name = 'r_thick_' + list(region.values())[0]
        thick_score[segment_name] = list(region.values())[1]
    x.update(thick_score)

    # LV function
    lv_fun = root[3][2][3].attrib
    lv_dict = dict()
    for key, value in lv_fun.items():
        new_key = 'r_' + key
        lv_dict[new_key] = value
    x.update(lv_dict)
    
    # ISCHEMIA
    # Reversibility
    rev = root[4][0][1].attrib
    x['rev_summedScore'] = rev['summedScore']
    rev_score = dict()
    for i in range(17):
        region = root[4][0][1][i].attrib
        segment_name = 'rev_' + list(region.values())[0]
        rev_score[segment_name] = list(region.values())[1]
    x.update(rev_score)
    
    LV_rev_per = dict()
    for i in range(4):
        region = root[4][0][2][i].attrib
        segment_name = 'rev_per_' + list(region.values())[0]
        LV_rev_per[segment_name] = list(region.values())[1]
    x.update(LV_rev_per)
    return x


def get_disp_img(img):
    scaled_image = (np.maximum(img, 0) / img.max()) * 255.0
    disp_img = np.uint8(scaled_image)
    return disp_img


def text_field(label, columns=None, **input_params):
    c1, c2 = st.columns((1, 2))

    # Display field name with some alignment
    c1.markdown("##")
    original_title = f'<p style="color:Green; font-size: 20px;">{label}</p>'
    c1.markdown(original_title, unsafe_allow_html=True)
    # c1.markdown(f'''
    # **:red[{label}]**''')

    # Sets a default key parameter to avoid duplicate key errors
    input_params.setdefault("key", label)

    # Forward text input parameters
    return c2.text_input(label, label_visibility='hidden',**input_params)


with st.expander(":blue[Upload Files]", expanded=True):
    # st.subheader("Upload Files")
    left_col, mid_col, right_col  = st.columns((2,2,2),gap="medium")
    
    with left_col:
        report_xml = st.file_uploader("Report File (XML)", type=['xml'])
    
    with mid_col:
        stress_dcm = st.file_uploader("Stress Polar Map (DICOM)", type=['dcm'])
    with right_col:
        rest_dcm = st.file_uploader("Rest Polar Map (DICOM)", type=['dcm'])

with st.container():

    left_col, mid_col, right_col  = st.columns((2,2,2),gap="medium")
    
    with left_col:
        # report_xml = st.file_uploader("Select Report File (XML)", type=['xml'])
        if report_xml is not None:
            xml = ElementTree.parse(report_xml)
            root = xml.getroot()
            # display_xml(xml)
            report_dict = read_xml_str_rst(root)
            report_df = pd.DataFrame.from_dict(report_dict, orient='index')
            
            st.subheader(':blue[Patient Info]')
            # 
            # st.dataframe(pt_info, use_container_width=True)
            # style = report_df.style.hide(axis='columns')
            # style.hide_columns()
            # st.write(style.to_html(), unsafe_allow_html=True)
            
            text_field('Patient Name', value=report_dict['name'])
            text_field('Patient ID', value=report_dict['id'])
            # text_field('Gender', value=report_dict['sex'])
            # text_field('Age', value=report_dict['age'])
            # text_field('Stress EF', value=report_dict['s_ef'])
            # text_field('Stress EDV', value=report_dict['s_edv'])
            # text_field('Stress ESV', value=report_dict['s_esv'])
            # text_field('Rest EF', value=report_dict['r_ef'])
            # text_field('Rest EDV', value=report_dict['r_edv'])
            # text_field('Rest ESV', value=report_dict['r_esv'])
            
            st.text('** Show important data from xml file **')
            pt_info = report_df.loc[['s_ef','s_edv', 's_esv', 'r_ef', 'r_edv', 'r_esv']]
            st.dataframe(pt_info, use_container_width=True)
            # style = pt_info.style.hide(axis='columns')
            # st.write(style.to_html(), use_container_width=True,
            #         unsafe_allow_html=True)
            
            
    
    with mid_col:
        # stress_dcm = st.file_uploader("Select Stress Polar Map (DICOM)", type=['dcm'])
        if stress_dcm is not None and report_xml is not None:
            stress_ds = dcmread(stress_dcm)
            st.subheader(':blue[Stress Polar Map]')
            display_polar(stress_ds)
            check_id(stress_ds, report_df)
            
    with right_col:
        # rest_dcm = st.file_uploader("Select Rest Polar Map (DICOM)", type=['dcm'])
        if rest_dcm is not None and report_xml is not None:
            rest_ds = dcmread(rest_dcm)
            st.subheader(':blue[Rest Polar Map]')
            display_polar(rest_ds)
            check_id(rest_ds, report_df)
    


with st.container():
    if report_xml is not None and stress_dcm is not None and rest_dcm is not None:
        if st.button('Proceed'):
            # result  = predictor.predict(model, stress_ds.pixel_array, rest_ds.pixel_array, report_df)
            
            st.subheader(':blue[Model Prediction]')
            st.text('** in progress **')
            st.write('result')
