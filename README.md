
# **Crime Analysis using KMeans Clustering & Random Forest**

A Django-based data analysis web application that explores and predicts crime patterns in India from 2001 to 2012. The project uses unsupervised learning (KMeans) for crime categorization and Random Forest Regression for future crime prediction, with detailed data visualizations built using Matplotlib.

## 📊 Features

- 🔍 Crime data analysis for categories: **Theft**, **Murder**, and **Rape**
- 📈 KMeans clustering for crime pattern grouping
- 🧠 Random Forest model for future crime rate prediction
- 📉 Interactive plots and visualizations using **Matplotlib**
- 📊 Dashboard-style web interface built using Django
- 🧼 Data preprocessing using Label Encoding and Min-Max Scaling

## 🛠️ Tech Stack

- **Backend:** Django
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib
- **ML Algorithms:** KMeans, Random Forest
- **Visualization:** Matplotlib (Agg backend for rendering in Django)
- **Language:** Python

## 📁 Folder Structure

```
crime_analysis/
├── templates/
├── static/
├── media/
├── app/                # Main Django app
│   ├── views.py
│   ├── urls.py
│   ├── models.py
│   └── ...
├── crime_data.csv      # Dataset used (2001–2012)
├── manage.py
└── requirements.txt
```

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Anvesh1909/Crime-Analysis-KMeans-RF.git
   cd Crime-Analysis-KMeans-RF
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Django App**
   ```bash
   python manage.py runserver
   ```

5. **Visit in Browser**
   ```
   http://127.0.0.1:8000/
   ```

## 📷 Sample Screenshots

> Add screenshots here once the app is running. Use tools like Lightshot or your browser's capture function.

## 📚 Dataset

- Source: [NCRB India Crime Data (2001–2012)](https://data.gov.in)
- Processed in CSV format for use in ML algorithms.

## 📌 Future Improvements

- Add crime heatmaps using GeoDjango and Leaflet
- Export results as PDF reports
- Add filtering by region or year range

## 🤝 Contributing

Feel free to fork and raise pull requests. Feedback and suggestions are welcome!

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
```

---

Let me know if you'd like me to generate a `requirements.txt`, sample screenshots, or help you create a live deployment for showcasing this project.
