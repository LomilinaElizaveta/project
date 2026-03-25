import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import wikipedia
wikipedia.set_lang('en')
from flask import Flask, render_template_string, request
model = pipeline('sentiment-analysis')
pd.set_option('display.max_columns', None)

# займемся датасетом

df = pd.read_csv('rotten_tomatoes.csv')
df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
df.rename(columns={'critic_score': 'rt_critic_score'}, inplace=True)
df.rename(columns={'people_score': 'rt_people_score'}, inplace=True)
df.rename(columns={'view_the_collection': 'collection'}, inplace=True)

df.drop(columns=['aspect_ratio', 'original_language',
                 'producer', 'release_date_(theaters)',
                 'release_date_(streaming)', 'runtime', 'sound_mix', 'synopsis', 'genre',
                 'writer', 'box_office_(gross_usa)', 'rating', 'total_ratings', 'type',
                 'collection', 'production_co', 'total_reviews', 'crew'], inplace=True)

df = df.dropna()

# сортирую в алфавитном порядке, чтобы на сайте было проще найти фильм

df = df.drop_duplicates(subset=['title', 'year'], keep='first')

df = df.sort_values('title')

# здесь я привожу в человеческий вид колонки

# округляю оценки
df['rt_people_score'] = (df['rt_people_score'] / 5).round(0) * 5
df['rt_critic_score'] = (df['rt_critic_score'] / 5).round(0) * 5

# получаю тональность консенсуса
df['consensus'] = df['consensus'].apply(str)
df['consensus'] = df['consensus'].apply(lambda x: model(x))
df['consensus'] = df['consensus'].apply(lambda x: round(x[0]['score'], 2))

# получаю нормальные числа

df['year'] = df['year'].astype(int)
df['rt_critic_score'] = df['rt_critic_score'].astype(int)
df['rt_people_score'] = df['rt_people_score'].astype(int)

# типизирую режиссеров
df['t_director'] = pd.factorize(df['director'])[0]

# делаю отдельный столбец для выбора на сайте
df['title_year'] = df['title'] + ' (' + df['year'].astype(str) + ')'


# обучим заранее модели

X1 = df.drop(columns=['id', 'title', 'title_year', 'year', 'link', 'rt_people_score', 'title_year',
                      'director'])
y1 = df['rt_critic_score']
X2 = df.drop(columns=['id', 'title', 'title_year', 'year', 'link', 'rt_critic_score', 'title_year',
                      'director'])
y2 = df['rt_people_score']

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=42)

X1_train, X1_val, y1_train, y1_val = train_test_split(
    X1_train, y1_train, test_size=0.3, random_state=42
    )

X2_train, X2_val, y2_train, y2_val = train_test_split(
    X2_train, y2_train, test_size=0.3, random_state=42
    )

k_best = -1
best_accuracy = 0

for k in range(1, 10):
    y_predicted = KNeighborsClassifier(n_neighbors=k).fit(X1_train, y1_train).predict(X1_val)
    val_accuracy = accuracy_score(y1_val, y_predicted)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        k_best = k
k_best = -1
best_accuracy = 0

for k in range(1, 10):
    y_predicted = KNeighborsClassifier(n_neighbors=k).fit(X2_train, y2_train).predict(X2_val)
    val_accuracy = accuracy_score(y2_val, y_predicted)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        k_best = k

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X1_train, y1_train)

knn2 = KNeighborsClassifier(n_neighbors=1)
knn2.fit(X2_train, y2_train)

# запишем в датасет

predictions1 = knn1.predict(X1)
df['predicted_rt_critic_score'] = predictions1
predictions2 = knn2.predict(X2)
df['predicted_rt_people_score'] = predictions2

# собственно, сайт.

app = Flask(__name__)

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {background: black; margin: 0; display: flex; min-height: 100vh; justify-content: center; align-items: center; font-family: Arial}
            .container {text-align: center}
            .dropdown-block {background: #333; color: white; padding: 15px 30px; cursor: pointer; margin-bottom: 20px}
            .dropdown-content {display: none; background: #444; min-width: 200px; margin-top: 5px}
            .dropdown-content a {color: white; display: block; cursor: pointer}
            .dropdown-content a:hover {background: #555}
            button {background: #444; color: white; padding: 10px 20px; border: none; cursor: pointer}
            select {padding: 10px; cursor: pointer; width: 250px; margin-bottom: 20px; background: #333; color: white; border: 1px solid #666}
        </style>
    </head>
    <body>
        <div class="container">
            <form action="/find" method="get">
                <select name="selected_title" required>
                    <option value="" disabled selected>Выберите фильм</option>
                    {% for title in titles %}
                    <option value="{{title}}">{{title}}</option>
                    {% endfor %}
                </select>
                <br>
                <button type="submit">Отправить</button>
            </form>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html, titles=df['title_year'].tolist())



@app.route('/find')
def find():
    selected_title = request.args.get('selected_title')
    a = df.loc[df['title_year'] == selected_title].iloc[0]
    link = a['link']
    rt_critic_score = a['rt_critic_score']
    predicted_rt_critic_score = a['predicted_rt_critic_score']
    rt_people_score = a['rt_people_score']
    predicted_rt_people_score = a['predicted_rt_people_score']
    director = a['director']
    combined_rt_critic_score = f"предсказанный - {predicted_rt_critic_score}, настоящий - {rt_critic_score}"
    combined_rt_people_score = f"предсказанный - {predicted_rt_people_score}, настоящий - {rt_people_score}"
    html_template = '''
           <!DOCTYPE html>
           <html>
           <head>
               <style>
                   body {margin: 0; padding: 0; background-color: black; color: white; font-family: Arial; font-size: 18px;
                   text-align: center}
                   h1 {font-size: 2.5em; padding: 20px; max-width: 90%; margin: 0 auto}
                   .container {padding-top: 50px}
                   .score-center {text-align: center; padding: 10px; margin-top: 20px; font-size: 1em}
               </style>
           </head>
           <body>
               <div class="container">
                   <h1>{{title}}</h1>
                   <h2>Режиссер: {{director}}</h2>
                   <div class="score-center">
                       Рейтинг критиков на <a href="{{link}}" target="_blank">rottentomatoes</a> (округленный): {{score}}
                   </div>
                   <div class="score-center">
                       Рейтинг зрителей на <a href="{{link}}" target="_blank">rottentomatoes</a> (округленный): {{people_score}}
                   </div>
               </div>
           </body>
           </html>
           '''
    return render_template_string(html_template, link=link, title=selected_title, score=combined_rt_critic_score, people_score=combined_rt_people_score, director=director)

if __name__ == '__main__':
    app.run(debug=False)

# идея с краулерами безнадежно провалилась, поскольку на ни википедия, ни imdb не поддавались ни попыткам притвориться браузером, ни разным прокси, ни запросам с различной частотой между ними.
# по этой же причине провалилась задумка с извлечением картинок. использование же API TMDB потребовало регистрации и личных данных, и показалось целесообразным к нему не прибегать