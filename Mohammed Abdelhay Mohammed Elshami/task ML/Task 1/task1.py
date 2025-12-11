
import os
import sys
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

# ---------- إعداد مسار الملف ----------
# يمكنك تعديل هذا المسار إلى مكان ملفك إن لزم
DEFAULT_FILENAME = "salaries(1).csv"

def find_csv_file(filename):
    # 1) لو المستخدم مرّر مسار كامل عبر CLI استخدمه
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    # 2) جرب المسار كما هو في المجلد الحالي
    if os.path.exists(filename):
        return os.path.abspath(filename)

    # 3) جرب المجلد الحالي + اسم الملف
    cand = os.path.join(os.getcwd(), filename)
    if os.path.exists(cand):
        return cand

    # 4) جرب على مسارات شائعة (يمكن تعديلها إذا لزم)
    possible = [
        os.path.join(os.getcwd(), "Task 1", filename),
        os.path.join("D:/task ML/Task 1", filename),
        os.path.join("D:\\task ML\\Task 1", filename),
        os.path.join(os.path.expanduser("~"), filename),
    ]
    for p in possible:
        if os.path.exists(p):
            return p

    return None

def main():
    # يمكن تمرير اسم/مسار الملف كأول باراميتر عند التشغيل
    input_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILENAME
    csv_path = find_csv_file(input_arg)

    if csv_path is None:
        print("خطأ: لم أجد ملف CSV. جرّب تمرير المسار كوسيط:")
        print(r"مثال: python task1.py 'D:\task ML\Task 1\salaries(1).csv'")
        sys.exit(1)

    print("قراءة الملف من:", csv_path)
    df = pd.read_csv(csv_path)

    print("\nعرض أول 5 صفوف:")
    print(df.head())

    # ---------- تجهيز البيانات ----------
    # التحقّق من وجود العمود المستهدف
    target_col = "has_diabetes"
    if target_col not in df.columns:
        print(f"\nخطأ: العمود '{target_col}' غير موجود في الملف.")
        print("أعمدة الملف هي:", df.columns.tolist())
        sys.exit(1)

    # Encode كل الأعمدة النصّية باستخدام LabelEncoder
    le = LabelEncoder()
    df_encoded = df.copy()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == object or df_encoded[col].dtype.name == 'category':
            # نتعامل مع القيم الفارغة قبل الencoding
            if df_encoded[col].isnull().any():
                print(f"تحذير: العمود '{col}' يحتوي على قيم NaN — سيتم تعبئتها بالقيمة 'missing'")
                df_encoded[col] = df_encoded[col].fillna("missing")
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # Features و target
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    print("\nأبعاد البيانات بعد المعالجة:")
    print("X:", X.shape)
    print("y:", y.shape)

    # ---------- تقسيم البيانات ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None
    )

    print(f"\nTrain size: {X_train.shape[0]} ، Test size: {X_test.shape[0]}")

    # ---------- تدريب النموذج ----------
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # ---------- التنبؤ والتقييم ----------
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy على مجموعة الاختبار: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # ---------- حفظ النموذج ----------
    model_filename = "model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": X.columns.tolist(),
            "target": target_col
        }, f)
    print(f"\nتم حفظ النموذج في: {os.path.abspath(model_filename)}")

    # ---------- رسم شجرة القرار ----------
    try:
        plt.figure(figsize=(12,8))
        plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())],
                  filled=True, rounded=True, fontsize=8)
        plt.title("Decision Tree")
        plt.tight_layout()
        # نحاول حفظ الصورة أيضاً
        plt.savefig("decision_tree.png", dpi=200)
        print(f"تم حفظ صورة شجرة القرار كـ: {os.path.abspath('decision_tree.png')}")
        # نحاول عرضها إذا كانت البيئة تدعم الـ GUI
        try:
            plt.show()
        except Exception:
            # في بيئة مثل بعض السيرفرات بدون عرض رسومي سيُرمى استثناء؛ نتجاهله لأننا حفظنا الصورة
            pass
    except Exception as e:
        print("حصل خطأ أثناء رسم شجرة القرار:", str(e))

if __name__ == "__main__":
    main()
