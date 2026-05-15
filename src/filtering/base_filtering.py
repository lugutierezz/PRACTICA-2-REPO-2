from feature_engine.selection import DropConstantFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class BaseFiltering:
    """
    Clase que encapsula el pipeline de seleccion de features en 3 etapas:
      1. Eliminar features constantes / cuasi-constantes (DropConstantFeatures)
      2. Eliminar features con baja variabilidad (DropConstantFeatures)
      3. Eliminar features poco relevantes (SelectFromModel + RandomForestClassifier)


    Sigue el patron fit/transform para poder ajustar en train y aplicar en test
    sin data leakage.
    """

    def __init__(self,
                 constant_tol=0.98,
                 variance_threshold=0.001,
                 probe_n_estimators=50,
                 probe_max_depth=10,
                 random_state=42):

        # Paso 1: eliminar features cuasi-constantes
        self.drop_constant = DropConstantFeatures(tol=constant_tol)

        # Paso 2: eliminar features con baja variabilidad
        self.variance_filter = VarianceThreshold(threshold=variance_threshold)

        # Paso 3: SelectFromModel con RandomForestClassifier.
        self.select = SelectFromModel(
            estimator=RandomForestClassifier( 
                n_estimators=probe_n_estimators,
                max_depth=probe_max_depth,
                random_state=random_state,
                n_jobs=-1
            ), #Calcula la importancia de cada feature usando un RandomForestClassifier

        threshold="mean"  
        )

    def fit(self, X_data, y_data):
        """
        Ajusta los 3 filtros secuencialmente sobre los datos de entrenamiento.
        Cada filtro aprende que features eliminar y guarda esa informacion
        para aplicarla luego en transform().
        """

        # Paso 1: fit + transform para que el paso 2 reciba datos ya filtrados
        self.drop_constant.fit(X_data)
        X_no_constant = self.drop_constant.transform(X_data)
        #Calculamos el numero de features eliminadas. 
        self.n_dropped_constant = X_data.shape[1] - X_no_constant.shape[1] #Calculamos el numero de features eliminadas. 

        # Paso 2: fit + transform para que el paso 3 reciba datos ya filtrados
        self.variance_filter.fit(X_no_constant)
        X_no_variance = self.variance_filter.transform(X_no_constant)

        support_variance = self.variance_filter.get_support() #Devuelve un vector de TRUE / FALSE en funciond e si la variable ha sido eliminada. Lo utilizamos para recuprar el nombre de las columnas. 
        self.cols_after_variance = X_no_constant.columns[support_variance].tolist()

        X_no_variance = pd.DataFrame(
            X_no_variance,
            columns=self.cols_after_variance,
            index=X_no_constant.index
        )

        self.n_dropped_variance = X_no_constant.shape[1] - X_no_variance.shape[1]

        # Paso 3: fit (el transform se hara cuando el usuario lo pida)
        self.select.fit(X_no_variance,y_data)
        X_final = self.select.transform(X_no_variance)

        support = self.select.get_support()
        self.selected_features = X_no_variance.columns[support].tolist()
        
        X_final = pd.DataFrame(
            X_final,
            columns=self.selected_features,
            index=X_no_variance.index
        )


        # Guardamos info para el resumen
        self.n_dropped_select = X_no_variance.shape[1] - X_final.shape[1] #Calculamos el numero de variables eliminadas en el ultimo paso. 
        self.n_features_initial = X_data.shape[1]
        self.n_features_final = X_final.shape[1]

    def transform(self, X_data):
        """
        Aplica los 3 filtros secuencialmente.
        Usa los parametros aprendidos en fit(), NO re-aprende nada.
        """
        X_out = self.drop_constant.transform(X_data)

        X_out = self.variance_filter.transform(X_out)
        X_out = pd.DataFrame(
            X_out,
            columns=self.cols_after_variance,
            index=X_data.index
        )

        X_out = self.select.transform(X_out)
        X_out = pd.DataFrame(
            X_out,
            columns=self.selected_features,
            index=X_data.index
        )
        return X_out

    def print_summary(self):
        """Imprime un resumen del pipeline de filtrado."""
        print("=" * 60)
        print("RESUMEN DEL PIPELINE DE FILTRADO")
        print("=" * 60)
        print(f"  Features iniciales:              {self.n_features_initial}")
        print(f"  Eliminadas cuasi-constantes:     -{self.n_dropped_constant}")
        print(f"  Eliminadas por variabilidad:      -{self.n_dropped_variance}")
        print(f"  Eliminadas por SelectModel:      -{self.n_dropped_select}")
        print(f"  Features seleccionadas finales:  {self.n_features_final}")
        print("=" * 60)