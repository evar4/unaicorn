.. code:: ipython3

    from sklearn.model_selection import train_test_split
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    df = pd.read_csv("train.csv")
    
    def get_title(name):
        if '.' in name:
            return name.split(',')[1].split('.')[0].strip()
        else:
            return 'Unknown'
    
    # A list with the all the different titles
    titles = sorted(set([x for x in df.Name.map(lambda x: get_title(x))]))
    
    
    # Normalize the titles
    def replace_titles(x):
        title = x['Title']
        if title in ['Capt', 'Col', 'Major']:
            return 'Officer'
        elif title in ["Jonkheer","Don",'the Countess', 'Dona', 'Lady',"Sir"]:
            return 'Royalty'
        elif title in ['the Countess', 'Mme', 'Lady']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        else:
            return title
        
    # Lets create a new column for the titles
    df['Title'] = df['Name'].map(lambda x: get_title(x))
    # train.Title.value_counts()
    # train.Title.value_counts().plot(kind='bar')
    
    # And replace the titles, so the are normalized to 'Mr', 'Miss' and 'Mrs'
    df['Title'] = df.apply(replace_titles, axis=1)
    
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna("S", inplace=True)
    df.drop("Cabin", axis=1, inplace=True)
    df.drop("Ticket", axis=1, inplace=True)
    df.drop("Name", axis=1, inplace=True)
    df.Sex.replace(('male','female'), (0,1), inplace = True)
    df.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
    df.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'), (0,1,2,3,4,5,6,7), inplace = True)
    
    
    predictors = df.drop(['Survived', 'PassengerId'], axis=1)
    target = df["Survived"]
    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
    
    
    randomforest = RandomForestClassifier()
    randomforest.fit(x_train, y_train)
    y_pred = randomforest.predict(x_val)
    
    filename = 'titanic_model.sav'
    pickle.dump(randomforest, open(filename, 'wb'))


.. parsed-literal::

    C:\Users\varna\AppData\Local\Temp\ipykernel_36484\793038460.py:40: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df['Age'].fillna(df['Age'].median(), inplace=True)
    C:\Users\varna\AppData\Local\Temp\ipykernel_36484\793038460.py:41: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df['Fare'].fillna(df['Fare'].median(), inplace=True)
    C:\Users\varna\AppData\Local\Temp\ipykernel_36484\793038460.py:42: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df['Embarked'].fillna("S", inplace=True)
    C:\Users\varna\AppData\Local\Temp\ipykernel_36484\793038460.py:46: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df.Sex.replace(('male','female'), (0,1), inplace = True)
    C:\Users\varna\AppData\Local\Temp\ipykernel_36484\793038460.py:46: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df.Sex.replace(('male','female'), (0,1), inplace = True)
    C:\Users\varna\AppData\Local\Temp\ipykernel_36484\793038460.py:47: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
    C:\Users\varna\AppData\Local\Temp\ipykernel_36484\793038460.py:47: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
    C:\Users\varna\AppData\Local\Temp\ipykernel_36484\793038460.py:48: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'), (0,1,2,3,4,5,6,7), inplace = True)
    C:\Users\varna\AppData\Local\Temp\ipykernel_36484\793038460.py:48: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'), (0,1,2,3,4,5,6,7), inplace = True)
    

