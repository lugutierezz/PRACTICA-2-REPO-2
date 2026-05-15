import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from skrub import GapEncoder, TextEncoder
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from skrub import SquashingScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import KNNImputer, SimpleImputer
from feature_engine.encoding import CountFrequencyEncoder

class BasePreprocess:

    def __init__(self, var_to_process, target):
        self.raw_predictors_vars = pd.read_excel(var_to_process)
        self.raw_predictors_vars = ( self.raw_predictors_vars
                                    .query("posible_predictora == 'si'")
                                    .variable
                                    .tolist())
        self.target_var = target
        self.poly = None

    def fit(self,data):
        
        # leemos dataframe
        df = pd.read_csv(data)


        # separamos x e y
        self.train_X_data = df[self.raw_predictors_vars]
        self.train_y_data = df[[self.target_var]]

        

        #####################################
        # tratamiento de nulls.
        ####################################
        size = self.train_X_data.shape[0]
 
        self.nulls_vars = ( (self.train_X_data.isnull().sum()/size)
                      .sort_values(ascending=False)
                      .to_frame(name="nulls_perc")
                      .reset_index() )
        
        print(self.train_X_data.isnull().sum().sum())
        
        
        # -------- MAYOR AL 98 % DE NULLS --------
        # descartamos aquellas vars cuyos nulls sean mayor al 98 %
        self.var_with_most_nulls = ( self.nulls_vars
                               .query("nulls_perc > 0.98")["index"]
                               .tolist() )
        
        self.train_X_data = self.train_X_data.drop(columns=self.var_with_most_nulls)
        
        

        # -------- MENOR AL 10 % DE NULLS --------
        self.nulls_10_perc = ( self.nulls_vars
                         .query("nulls_perc < 0.10")["index"]
                         .tolist() )
        
        self.categoric_vars = ( self.train_X_data
                        .loc[:, ~self.train_X_data.columns.isin(self.var_with_most_nulls)]
                        .select_dtypes(include="object")
                        .columns.tolist() )
        
        self.numeric_vars = (self.train_X_data
            .loc[:, ~self.train_X_data.columns.isin(self.var_with_most_nulls)]
            .select_dtypes(include="number")
            .columns.tolist() )

        # Variables numéricas con pocos nulos utilizamso KNNImputer
        self.numeric_less10_nulls = [col for col in self.nulls_10_perc if col in self.numeric_vars]

        if self.numeric_less10_nulls:
            self.knn_imputer = KNNImputer(n_neighbors=5)
            self.knn_imputer.fit(self.train_X_data[self.numeric_less10_nulls])
            self.train_X_data[self.numeric_less10_nulls] = self.knn_imputer.transform(
                            self.train_X_data[self.numeric_less10_nulls]
                        )

        # Variables categóricas con pocos nulos utilizamos SimpleImputer con estrategia de moda
        self.categoric_less10_nulls = [col for col in self.nulls_10_perc if col in self.categoric_vars]

        if self.categoric_less10_nulls:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            self.cat_imputer.fit(self.train_X_data[self.categoric_less10_nulls])
            self.train_X_data[self.categoric_less10_nulls] = self.cat_imputer.transform(
                self.train_X_data[self.categoric_less10_nulls]
            )

        # -------- ENTRE 10 % Y 98 % DE NULLS --------
        self.nulls_more_10_perc = ( self.nulls_vars
                              .query("nulls_perc >= 0.10 and nulls_perc <= 0.98")["index"]
                              .tolist() )

        # Variables numéricas -> imputar con -1
        self.numeric_vars_more_10 = [col for col in self.nulls_more_10_perc if col in self.numeric_vars]

        if self.numeric_vars_more_10:
            self.imp_num_p10 = SimpleImputer(strategy="constant", fill_value=-1)
            self.imp_num_p10.fit(self.train_X_data[self.numeric_vars_more_10])
            self.train_X_data[self.numeric_vars_more_10] = self.imp_num_p10.transform(
                self.train_X_data[self.numeric_vars_more_10]
            )

        # Variables categóricas -> imputar con "DESCONOCIDO"

        self.categoric_vars_more_10 = [col for col in self.nulls_more_10_perc if col in self.categoric_vars]
        
        if self.categoric_vars_more_10:
            self.imp_cat_p10 = SimpleImputer(strategy="constant",fill_value="DESCONOCIDO")
            self.imp_cat_p10.fit(self.train_X_data[self.categoric_vars_more_10])
            self.train_X_data[self.categoric_vars_more_10] = self.imp_cat_p10.transform(
                self.train_X_data[self.categoric_vars_more_10]
            )

        print(self.train_X_data.isnull().sum().sum())

        
        ###########################################
        # Extraer mes y año de variables temporales
        ###########################################
        self.train_X_data['earliest_cr_line'] = pd.to_datetime(self.train_X_data['earliest_cr_line'])
        self.train_X_data['earliest_cr_line_year'] = self.train_X_data['earliest_cr_line'].dt.year
        self.train_X_data['earliest_cr_line_month'] = self.train_X_data['earliest_cr_line'].dt.month.astype(str)

        ###################################
        # Procesamos variables categóricas
        ###################################

        # vemos las categoricas ordenas por cardinalidad.
        categoric_vars_cardinality = ( self.train_X_data[self.categoric_vars]
                              .nunique()
                              .sort_values(ascending=False)
                              .to_frame(name="cardinality")
                              .reset_index())
        
        # -------- VARIABLES CPN CARDINALIDAD MENOR A 50 --------
        self.low_cardinality_vars_names = categoric_vars_cardinality.query("cardinality <= 50")["index"].tolist()

        self.encoder = CountFrequencyEncoder(encoding_method="frequency")
        self.encoder.fit(self.train_X_data[self.low_cardinality_vars_names])

        # --------- VARIABLES CON CARDINALIDAD MAYOR A 50 (EMP_TITLE) --------

        if "emp_title" in self.train_X_data.columns:
            self.gap_encoder = GapEncoder(n_components=10, random_state=0)
            self.gap_encoder.fit(self.train_X_data["emp_title"])

        # --------- VARIABLES DE TEXTO (DESC) --------

        self.train_X_data['desc_formated'] = np.where(
            self.train_X_data['desc'] == 'DESCONOCIDO',
            'DESCONOCIDO',
            self.train_X_data['desc'].str.split('> ').str[1].str.split('<br>').str[0]
            )
        
        self.text_enc_desc = TextEncoder(model_name='intfloat/e5-small-v2', n_components=20)
        self.text_enc_desc.fit(self.train_X_data['desc_formated'])

        

        ###################################
        # transformación variables numéricas
        ###################################

        self.numeric_vars = ( self.train_X_data
                            .loc[:, ~self.train_X_data.columns.isin(self.var_with_most_nulls)]
                            .select_dtypes(include='number')
                            .columns.tolist() )
        
        numeric_data = self.train_X_data.select_dtypes(include="number")
        
        self.minmax = MinMaxScaler()
        self.minmax.fit(numeric_data)

        

    
    def transform(self, data):

        df = pd.read_csv(data)
        self.X_data = df[self.raw_predictors_vars]
        self.y_data = df[[self.target_var]]

        self.categoric_vars = ( self.X_data
                .loc[:, ~self.X_data.columns.isin(self.var_with_most_nulls)]
                .select_dtypes(include="object")
                .columns.tolist() )
        
        self.numeric_vars = (self.X_data
            .loc[:, ~self.X_data.columns.isin(self.var_with_most_nulls)]
            .select_dtypes(include="number")
            .columns.tolist() )

        #####################################
        # tratamiento de nulls.
        ####################################
        
        # -------- MAYOR AL 98 % DE NULLS --------
        # descartamos aquellas vars cuyos nulls sean mayor al 98 %
        self.var_with_most_nulls = ( self.nulls_vars
                               .query("nulls_perc > 0.98")["index"]
                               .tolist() )
        
        self.X_data = self.X_data.drop(columns=self.var_with_most_nulls)
        
        

        # -------- MENOR AL 10 % DE NULLS --------

        # Variables numéricas con pocos nulos utilizamso KNNImputer

        if self.numeric_less10_nulls:
            self.X_data[self.numeric_less10_nulls] = self.knn_imputer.transform(
                            self.X_data[self.numeric_less10_nulls]
                        )

        # Variables categóricas con pocos nulos utilizamos SimpleImputer con estrategia de moda

        if self.categoric_less10_nulls:
            self.X_data[self.categoric_less10_nulls] = self.cat_imputer.transform(
                self.X_data[self.categoric_less10_nulls]
            )

        # -------- ENTRE 10 % Y 98 % DE NULLS --------

        # Variables numéricas -> imputar con -1
        self.numeric_vars_more_10 = [col for col in self.nulls_more_10_perc if col in self.numeric_vars]

        if self.numeric_vars_more_10:
            self.X_data[self.numeric_vars_more_10] = self.imp_num_p10.transform(
                self.X_data[self.numeric_vars_more_10]
            )

        # Variables categóricas -> imputar con "DESCONOCIDO"

        self.categoric_vars_more_10 = [col for col in self.nulls_more_10_perc if col in self.categoric_vars]
        
        if self.categoric_vars_more_10:
            self.X_data[self.categoric_vars_more_10] = self.imp_cat_p10.transform(
                self.X_data[self.categoric_vars_more_10]
            )

        ###########################################
        # Extraer mes y año de variables temporales
        ###########################################
        self.X_data['earliest_cr_line'] = pd.to_datetime(self.X_data['earliest_cr_line'])
        self.X_data['earliest_cr_line_year'] = self.X_data['earliest_cr_line'].dt.year
        self.X_data['earliest_cr_line_month'] = self.X_data['earliest_cr_line'].dt.month.astype(str)
        

        ###################################
        # Procesamos variables categóricas
        ###################################
        
        # -------- VARIABLES CON CARDINALIDAD MENOR A 50 --------
        X_low_cardinality = self.encoder.transform(self.X_data[self.low_cardinality_vars_names])
        X_low_cardinality = pd.DataFrame(
                            X_low_cardinality,
                            columns=self.low_cardinality_vars_names,
                            index=self.X_data.index)
        
        # --------- VARIABLES CON CARDINALIDAD MAYOR A 50 (EMP_TITLE) --------
        if "emp_title" in self.X_data.columns:
            X_gap_encoder = self.gap_encoder.transform(self.X_data["emp_title"])
            X_gap_encoder = pd.DataFrame(X_gap_encoder, 
                                      columns=self.gap_encoder.
                                      get_feature_names_out(["emp_title"]), 
                                      index=self.X_data.index)
        
        # --------- VARIABLES DE TEXTO (DESC) --------

        self.X_data['desc_formated'] = np.where(
            self.X_data['desc'] == 'DESCONOCIDO',
            'DESCONOCIDO',
            self.X_data['desc'].str.split('> ').str[1].str.split('<br>').str[0]
            )
        
        X_text_desc = self.text_enc_desc.transform(self.X_data["desc_formated"])
        

        ###################################
        # transformación variables numéricas
        ###################################

        
        self.numeric_data = self.X_data.select_dtypes(include="number")

        X_minmax = self.minmax.transform(self.numeric_data)
        X_minmax = pd.DataFrame(
        X_minmax,
        columns=self.numeric_data.columns,
        index=self.X_data.index)

        #####################################
        # Nuevas features cruzadas
        #####################################
    
        # range FICO: calculamos la diferencia entre el FICO máximo y mínimo.
        self.X_data["range_fico"] = (
        self.X_data["fico_range_high"] -
        self.X_data["fico_range_low"])


        # mean FICO: calculamos la media entre el FICO máximo y mínimo.
        self.X_data["mean_FICO"] = (
            self.X_data["fico_range_high"] +
            self.X_data["fico_range_low"]) / 2
        

        # fico_category: categorizamos el FICO medio en 4 categorías (0, 1, 2, 3) 
        self.X_data["fico_category"] = pd.cut(
            self.X_data["mean_FICO"],
            bins=[300, 580, 670, 740, 850],
            labels=[0, 1, 2, 3])
        
        # ratio_recent_delinquency: calculamos el ratio entre el número de delinquencies en los últimos 2 años y el número total de cuentas con delinquency.
        self.X_data["ratio_recent_delinquency"] = (
            self.X_data["delinq_2yrs"] /
            (self.X_data["acc_now_delinq"] + 1))

        

        #############################
        # Concatenar datos
        ##############################

        X_train = pd.concat([
            X_minmax,
            X_low_cardinality,
            X_gap_encoder,
            X_text_desc,
            self.X_data["range_fico"],
            self.X_data["mean_FICO"],
            self.X_data["fico_category"],
            self.X_data["ratio_recent_delinquency"],
        ], axis=1)
        
        # transformar y_data
        y_data_out = self.y_data != 'Fully Paid'
        return X_train, y_data_out