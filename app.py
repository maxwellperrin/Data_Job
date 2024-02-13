import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import datetime


# Import des fichiers csv.
df = pd.read_csv("database.csv")
        
df['date_publi'] = pd.to_datetime(df['date_publi'])
df['date_publi'] = df['date_publi'].dt.date


# Définir le titre et le logo de la page
st.set_page_config(
    page_title="Data Job Platform",
    page_icon=":dizzy:",
    layout='wide'
)

st.title("Data Job Platform")
st.write('')
st.write('')


all_ids = []
for ids in df['id'] : 
    all_ids.append(ids)


def compare_skills_with_tokens(competence):

    competence = [skill.lower() for skill in competence]
    liste_compe = {}

    if len(competence) == 0 :
        return all_ids

    else :
        for skill in competence:
            liste_compe[skill] = []

        # Remplissez chaque liste avec les IDs correspondants
        for skill, id_list in liste_compe.items():
            liste_compe[skill] = [id for id, token in zip(df['id'], df['competences']) if skill in token]

        # Utilisez l'intersection pour trouver les IDs communs
        common_ids_set = set(liste_compe[competence[0]])

        for skill in competence:
            common_ids_set &= set(liste_compe[skill])

        # Convertissez l'ensemble résultant en liste
        common_ids_list = list(common_ids_set)

        return common_ids_list


today = date.today()
week_ago = today - timedelta(days=7)


# ---------------------------------------------------------------------------

if __name__ == '__main__':

    with st.sidebar:
        st.image("Capture d'écran 2024-02-05 145718 2.png")
        st.write('')
        search_type = st.radio("", ("Page d'accueil", 'Mon compte', 'Mes offres'))
        st.divider()

        with st.expander("**Se connecter**") :
            adresse_mail = st.text_input('Email')
            mot_de_passe = st.text_input('Mot de passe', type='password')
        st.divider()
        with st.expander("Nos partenaires") :
            st.image("logo-hellowork.svg", width= 120)
            st.write('')
            st.image("logo-pole-emploi.svg", width= 120) 
            st.image("Group 3653.png")
            st.image("talentify.svg")

    if search_type == "Page d'accueil":

        col1, col2 = st.columns(2)

        with col1 :
            competences_tech_1 = [
            'SQL', 'Python', 'Power BI', 'Machine Learning', 
            'R', 'Java', 'C++', 'C#', 'Dataiku', 'JavaScript',
            'HTML', 'CSS', 'React', 'Angular', 'Vue.js',
            'Data Analysis', 'Data Science', 'Deep Learning',
            'Statistical Analysis', 'Big Data', 'Hadoop', 'Spark',
            'Database Management', 'MySQL', 'PostgreSQL', 'MongoDB', 'SQLite',
            'Data Visualization', 'Tableau', 'Matplotlib', 'Seaborn',
            'Excel', 'Pandas', 'NumPy', 'SciPy', 'Scikit-learn',
            'ETL', 'Data Warehousing', 'Business Intelligence',
            'Version Control', 'Git', 'GitHub', 'GitLab',
            'APIs', 'RESTful', 'SOAP', 'GraphQL',
            'Web Development', 'Frontend', 'Backend', 'Full Stack',
            'Linux', 'Unix', 'Shell Scripting', 'Bash',
            'Cloud Computing', 'AWS', 'Azure', 'Google Cloud Platform',
            'Docker', 'Kubernetes', 'Containerization',
            'DevOps', 'Continuous Integration', 'Continuous Deployment',
            'Agile', 'Scrum', 'Kanban'
                                    ]
            compétence_1 = st.multiselect(
                        'Entrez vos compétences :',
                       competences_tech_1
                        )
            
            df_link_id_competence = df
            if (len(compétence_1) > 0) : 
                df_link_id_competence = df[df['competences_matched'].apply(lambda x: len(x) > 0)]

        
        with col2 : 
            ville = st.selectbox("Où ?", ["France"] + list(df_link_id_competence.ville.unique()))

            if (ville == 'France') and (len(compétence_1) == 0) : 
                df_link_id_competence = df_link_id_competence

            else :
                df_link_id_competence = df_link_id_competence[df_link_id_competence['ville'] == ville]

        # Use st.beta_columns to arrange checkboxes on the same line
        col1, col2 = st.columns(2)

        with col1 : 
            option = st.selectbox(
            'Dates',
            ("", "Dernières 24 heures", 'Depuis une semaine'))

        with col2 : 
            contrat = st.multiselect('Contrat', df['contrat'].unique())
            if len(contrat) > 0 : 
                df_link_id_competence = df_link_id_competence[df_link_id_competence['contrat'].isin(contrat)]

            else: 
                df_link_id_competence = df_link_id_competence        
        st.write(' ---------------------------- ')

        if option == 'Dernières 24 heures' :
            df_link_id_competence = df_link_id_competence[(df_link_id_competence['date_publi'] >= today)]
        elif option ==  'Depuis une semaine':
            df_link_id_competence = df_link_id_competence[(df_link_id_competence['date_publi'] >= week_ago)] 

            # ----------------------------
        else :
            df_link_id_competence = df_link_id_competence
        st.write(f'**{len(df_link_id_competence)} offres**')

    # Afficher chaque information dans un expander
        for index, row in df_link_id_competence.iterrows():
            with st.expander(f"Offre {index + 1} - {row['offre']}") :
                st.write(f"**Contrat** : {row['contrat']}")
                st.write(f"**Date de publication** : {row['date_publi']}")
                st.write(f"**Entreprise** : {row['entreprise']}")
                st.write(f"**Ville** : {row['ville']}")
                st.write(f"**Description** : {row['description']}")
                st.link_button("**Postuler**", f"{row['url_page']}", help=None, type="secondary", disabled=False, use_container_width=False)


    if search_type == 'Mon compte':
        from PIL import Image, ImageOps

        try:
            image_path = 'photo_profil.png'
            original_image = Image.open(image_path)

            # Create a thumbnail with the desired size
            original_image.thumbnail((125, 125))

            # Display the resized image
            st.image(original_image, use_column_width=False)

            col10, col11, col12, col13, col14, col15 = st.columns(6)
            with col15:
                identitad = pd.read_csv('identité_user.csv')
                st.write(f'**Hello {identitad["prenom"][0]} {identitad["nom"][0]}**')

        except:
            pass
        
        col_nom, col_prenom = st.columns(2)

        with col_nom :
            nom = st.text_input('Nom')
            adresse_mail = st.text_input('Email', key='aemc')

        with col_prenom :
            prénom = st.text_input('Prénom')
            mot_de_passe = st.text_input('Mot de passe', type='password', key='mdpmc')

        if len(nom) > 0 or len(prénom) > 0:
            data = [{'nom': nom, 'prenom': prénom}]
            identité = pd.DataFrame(data)
            identité.to_csv('identité_user.csv', index=False)
        
# CV
        uploaded_files = st.file_uploader("J'ajoute mon CV", accept_multiple_files=True, key='zferferferf') 
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read() 

# PHOTO
        from PIL import Image

        photo_oui = st.selectbox(
            'Tu veux ajouter une photo ?',
            ("", "Oui", 'Non', "Prendre une photo"))

        if photo_oui == 'Prendre une photo':
            picture = st.camera_input("")
            if picture is not None:
                # To read image file buffer as a PIL Image:
                img = Image.open(picture)
                img.save("photo_profil.png")
            
        elif photo_oui == 'Oui':
            pictures = st.file_uploader("J'ajoute ma photo", accept_multiple_files=True, key='egefeezfr')  

            if pictures:
                for picture in pictures:
                    img = Image.open(picture)
                    img.save("photo_profil.png")

        else :
            pass

        st.divider()
        df_utilisateur = df

        competences_tech = [
            'SQL', 'Python', 'Power BI', 'Machine Learning', 
            'R', 'Java', 'C++', 'C#', 'Dataiku', 'JavaScript',
            'HTML', 'CSS', 'React', 'Angular', 'Vue.js',
            'Data Analysis', 'Data Science', 'Deep Learning',
            'Statistical Analysis', 'Big Data', 'Hadoop', 'Spark',
            'Database Management', 'MySQL', 'PostgreSQL', 'MongoDB', 'SQLite',
            'Data Visualization', 'Tableau', 'Matplotlib', 'Seaborn',
            'Excel', 'Pandas', 'NumPy', 'SciPy', 'Scikit-learn',
            'ETL', 'Data Warehousing', 'Business Intelligence',
            'Version Control', 'Git', 'GitHub', 'GitLab',
            'APIs', 'RESTful', 'SOAP', 'GraphQL',
            'Web Development', 'Frontend', 'Backend', 'Full Stack',
            'Linux', 'Unix', 'Shell Scripting', 'Bash',
            'Cloud Computing', 'AWS', 'Azure', 'Google Cloud Platform',
            'Docker', 'Kubernetes', 'Containerization',
            'DevOps', 'Continuous Integration', 'Continuous Deployment',
            'Agile', 'Scrum', 'Kanban'
                                    ]
        compétence = st.multiselect(
                        'Entrez vos compétences :',
                       competences_tech
                        )
        if (len(compétence) > 0) : 
            df_utilisateur = df[df['competences_matched'].apply(lambda x: len(x) > 0)]
     

        else: 
            df_utilisateur = df
        col1, col2, col3 = st.columns(3)

        with col1 : 
            option = st.date_input("Chercher des offres avant :", today)
            df_utilisateur = df_utilisateur[(df_utilisateur['date_publi'] >= option)]
        
        with col2: 
            df = df.sort_values(by = 'ville', ascending=True)
            localisation = st.multiselect('Localisation', df['ville'].unique())

            if (len(localisation) > 0) and (localisation[0] != 'France') : 
                df_utilisateur = df_utilisateur[df_utilisateur['ville'].isin(localisation)]
                    
            else: 
                df_utilisateur = df_utilisateur
        
        with col3: 
            contrat = st.multiselect('Contrat', df['contrat'].unique())
            if len(contrat) > 0 : 
                df_utilisateur = df_utilisateur[df_utilisateur['contrat'].isin(contrat)]

            else: 
                df_utilisateur = df_utilisateur
        
        
        txt = st.text_area('Description de votre profil')
        st.write(f'You wrote {len(txt)} characters.')
        df_utilisateur.to_csv('database_user.csv', index=False)
    
    if search_type == 'Mes offres':

        # ----------- PROCESSING ML
        df_user = pd.read_csv('database_user.csv')
        col_drop = ['pays', 'description', 'url_page', 'competences']
        df_user = df_user.drop(columns=col_drop)
        df_user = df_user[df_user['entreprise'].notna()]
        df_user = df_user[df_user['lat'].notna()]

        df_user['competences_matched'] = df_user['competences_matched'].str.replace('[', '').str.replace(']', '')
        df_user['competences_matched'] = df_user['competences_matched'].str.split(',').apply(lambda x: [item.strip("' ") for item in x])
        # competences_list = df_user['competences_matched'].explode().tolist()
        df_user['competences_matched'] = df_user['competences_matched'].apply(lambda x : set(x))
        df_user['competences_matched'] = df_user['competences_matched'].apply(lambda x : list(x))

        df_machine_learning1 = df_user.explode('competences_matched')
        competences_dummies1 = df_machine_learning1['competences_matched'].str.get_dummies(sep=', ')
        df_machine_learning1 = pd.concat([df_machine_learning1, competences_dummies1], axis=1)
        result = df_machine_learning1.groupby('id')[df_machine_learning1.iloc[:, 14:].columns].sum().reset_index()
        df_choix = df_user.merge(result, how='inner', left_on='id', right_on='id')
        col_drop = ['id', 'competences_matched']
        df_choix = df_choix.drop(columns=col_drop)

        # --------------------------
        df2 = pd.read_csv('database_user.csv')

        col_drop = ['pays', 'description', 'url_page', 'competences']
        df2 = df2.drop(columns=col_drop)
        df2 = df2[df2['entreprise'].notna()]
        df2 = df2[df2['lat'].notna()]

        df2['competences_matched'] = df2['competences_matched'].str.replace('[', '').str.replace(']', '')
        df2['competences_matched'] = df2['competences_matched'].str.split(',').apply(lambda x: [item.strip("' ") for item in x])
        # competences_list = df2['competences_matched'].explode().tolist()
        df2['competences_matched'] = df2['competences_matched'].apply(lambda x : set(x))
        df2['competences_matched'] = df2['competences_matched'].apply(lambda x : list(x))

        df_machine_learning = df2.explode('competences_matched')
        competences_dummies = df_machine_learning['competences_matched'].str.get_dummies(sep=', ')
        df_machine_learning = pd.concat([df_machine_learning, competences_dummies], axis=1)
        # columns_to_sum = ['agile', 'angular', 'apis', 'aws', 'azure', 'backend', 'bash', 'devops', 'docker', 'etl', 'excel', 'frontend', 'git', 'github', 'gitlab', 'hadoop', 'html', 'java', 'javascript', 'kanban', 'linux', 'matplotlib', 'mongodb', 'mysql', 'numpy', 'postgresql', 'python', 'r', 'react', 'scrum', 'seaborn', 'spark', 'sql', 'sqlite', 'tableau', 'unix']
        result = df_machine_learning.groupby('id')[df_machine_learning.iloc[:, 14:].columns].sum().reset_index()
        df_ML = df2.merge(result, how='inner', left_on='id', right_on='id')
        col_drop = ['id', 'competences_matched']
        df_ML = df_ML.drop(columns=col_drop)

# ----------------------------------------------------------------------



        df_utilisateur = pd.read_csv('database_user.csv')
        st.write(f'**{len(df_utilisateur)} offres** en fonction de votre profil')
        for index, row in df_utilisateur.iterrows():
            with st.expander(f"Offre {index + 1} - {row['offre']}") :
                st.write(f"**Contrat** : {row['contrat']}")
                st.write(f"**Date de publication** : {row['date_publi']}")
                st.write(f"**Entreprise** : {row['entreprise']}")
                st.write(f"**Ville** : {row['ville']}")
                st.write(f"**Description** : {row['description']}")
                st.link_button("**Postuler**", f"{row['url_page']}", help=None, type="secondary", disabled=False, use_container_width=False) 
    
        st.divider()

        nombre_voisin = st.slider("**Combien d'offres ?**", 0, 25, 5)

        from sklearn.compose import make_column_selector as selector
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import make_pipeline
        from sklearn.neighbors import NearestNeighbors

        X = df_ML

        selector_num_cols = selector(dtype_exclude=object)
        selector_cat_cols = selector(dtype_include=object)

        num_cols = selector_num_cols(X)
        cat_cols = selector_cat_cols(X)

        num_preprocessor = StandardScaler()
        cat_preprocessor = OneHotEncoder()
        preprocessor = ColumnTransformer(
                    [
                        ("OneHotEncoder", cat_preprocessor, cat_cols),
                        ("StandardScaler", num_preprocessor, num_cols),
                    ]
                )
        neigh = NearestNeighbors(n_neighbors=nombre_voisin)
        X_scaled = preprocessor.fit_transform(X)
        neigh.fit(X_scaled)
        cible = df_choix
            
        user = preprocessor.transform(cible) 
        array = neigh.kneighbors(user)

        list_chiffre = []
        index = []

        for i in array[0] :
            for y in i :
                list_chiffre.append(y)
        list_chiffre.sort()
        list_chiffre = list_chiffre[len(array[0]):]


        for i in range(len(list_chiffre[:nombre_voisin])):
            index.append(df_ML.iloc[array[1][0][i]].name)

        df = pd.read_csv('database.csv')
        df_propal = df[df.index.isin(index)]

        st.write(f'**Ces offres pourraient vous plaire**')
        for index, row in df_propal.iterrows():
            with st.expander(f"Offre {index + 1} - {row['offre']}") :
                st.write(f"**Contrat** : {row['contrat']}")
                st.write(f"**Date de publication** : {row['date_publi']}")
                st.write(f"**Entreprise** : {row['entreprise']}")
                st.write(f"**Ville** : {row['ville']}")
                st.write(f"**Description** : {row['description']}")
                st.link_button("**Postuler**", f"{row['url_page']}", help=None, type="secondary", disabled=False, use_container_width=False) 

