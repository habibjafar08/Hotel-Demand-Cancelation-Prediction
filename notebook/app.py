import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pycaret.classification import load_model

@st.cache_resource() 
def get_model(): 
    return load_model("Deployment20052024") 

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    model=get_model()
    st.title("Hotel Cancelation Prediction App")
    add_selectbox=st.selectbox("Pilih Cara Anda Untuk Melakukan Prediksi?", ("Online" , "Batch"))
    # menambahkan keterangan pada sidebar
    st.sidebar.info('Aplikasi ini digunakan untuk memprediksi potensi cancelation pada pengunjung hotel')
    
    # Menambahkan title
    
    if add_selectbox=="Online":
        customer_type=st.selectbox('Custome Type:',('Transient-Party','Transient','Contract','Group'))
        reserved_room_type=st.selectbox('Reserved Room Type:',('A','E','D','F','B','G','C','H','L','P'))
        previous_cancellati = st.number_input('Previous Cancellation:',min_value=0,max_value=26,step=1)
        booking_changes=st.number_input('Booking Change: ',min_value=0,max_value=21,step=1)
        days_in_waiting_list=st.number_input('Days In Waiting List:',min_value=0,max_value=391,step=1)
        required_car_parking_spaces=st.number_input('Required Car Parking Spaces:',min_value=0,step=1)
        total_of_special_requests=st.number_input('Total Of Special Requests:',min_value=0,max_value=5,step=1)

        input_df=pd.DataFrame([
            {
                'previous_cancellations': previous_cancellati,
                'booking_changes': booking_changes,
                'days_in_waiting_list': days_in_waiting_list,
                'customer_type': customer_type,
                'reserved_room_type': reserved_room_type,
                'required_car_parking_spaces': required_car_parking_spaces,
                'total_of_special_requests': total_of_special_requests, 
            }
        ])

        output = ""

        # Make a prediction 
        if st.button("Predict"):
            output = model.predict(input_df)
            if (output[0] == 0):
                output = 'The person is not cancel'
            else:
                output = 'The person is Cancel'

        # Show prediction result
        st.success(output)    

    if add_selectbox=="Batch":
        file_Upload=st.file_uploader("Upload CSV file untuk memprediksi", type=["csv"])
        

        if file_Upload is not None:
            data=pd.read_csv(file_Upload)

            #select kolom
            data=data[[
                'previous_cancellations',
                'booking_changes',
                'days_in_waiting_list',
                'customer_type',
                'reserved_room_type',
                'required_car_parking_spaces',
                'total_of_special_requests',
            ]]

            #Prediction
            data['Prediction']= np.where(model.predict(data)==1, 'Cancel','Not Cancel')

            # menampilkan hasi prediksi
            st.write(data)

            #Menambahkan button download 
            st.download_button(
                "Press button untuk mendownload",convert_df(data),"Hasil prediksi.csv","text/csv",key='download-csv')
    
if __name__ == '__main__':
    main()

