import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User
from django.contrib import messages
import numpy as np
from django.http import HttpResponse
from gtts import gTTS
from tensorflow.keras.preprocessing import image
import cv2

import tensorflow as tf
# Loading the model
model = tf.keras.models.load_model("./model/Neww.h5")
model1 = tf.keras.models.load_model("./model/Lung.h5")

# Function to capture an image from the camera
def capture_image():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Camera Feed', frame)  # Show the camera feed in a window

        # Break the loop and capture the current frame when a key is pressed
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

# Function to resize the captured image
def resize_image(img, target_size=(224, 224)):
    return cv2.resize(img, target_size)

def home(request):
    contexts = {}
    msg = ''
    Duration = ''
    if request.method == 'POST':
        if 'open-camera' in request.POST:
            # Capture an image from the camera
            captured_frame = capture_image()
            resized_frame = resize_image(captured_frame)

            # Convert the captured frame to an image
            img = image.array_to_img(resized_frame, data_format="channels_last", scale=False)

            # Preprocess the image for the model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.

            # Use the model to predict the class of the image
            prediction = model.predict(img_array)

            # Get the predicted class label
            predicted_class = np.argmax(prediction[0])

            # Define your disease labels here
            namesDiseases = ["HairLoss", "Acne and Rosacea", "Actinic Keratosis Basal Cell Carcinoma",
                             "Melanoma Skin Cancer Nevi and Moles", "Vasculties"]

            # Create a dictionary to map disease labels to categories and durations
            disease_info = {
                "hairloss": {
                    "category": "Normal",
                    "duration": "The average duration of treatment is 6 months to 1 year."
                },
                "acne and rosacea": {
                    "category": "Normal",
                    "duration": "The average duration of treatment is 5 to 6 months."
                },
                "actinic keratosis basal cell carcinoma": {
                    "category": "Dangerous",
                    "duration": "The average duration of treatment is 3 months."
                },
                "melanoma skin cancer nevi and moles": {
                    "category": "Dangerous",
                    "duration": "The average duration of treatment is 6 to 12 months."
                },
                "vasculties": {
                    "category": "Dangerous",
                    "duration": "The average duration of treatment is several months to a few years."
                }
            }

            predicted_label = namesDiseases[predicted_class]

            Proba = prediction[0][predicted_class] * 100

            search_str = predicted_label.lower()

            if search_str in disease_info:
                category = disease_info[search_str]["category"]
                duration = disease_info[search_str]["duration"]

                if category == "Dangerous":
                    msg = "It's a DANGEROUS DISEASE, you need to see a doctor. " \
                          "It's important to recognize that some diseases can be very dangerous " \
                          "and require immediate attention from a medical professional. Delaying " \
                          "a doctor's visit for a serious illness can worsen the condition and " \
                          "potentially lead to life-threatening consequences. It's always better " \
                          "to err on the side of caution and seek medical advice as soon as possible, " \
                          "especially if you notice any unusual symptoms or changes in your health. " \
                          "Remember, your health is your most valuable asset, and taking care of " \
                          "it should be a top priority."
                elif category == "Normal":
                    msg = "It is NOT A DANGEROUS DISEASE, you can take your time. " \
                          "If you have been diagnosed with a disease that is not considered serious, " \
                          "it is important to still take it seriously and not ignore it. While it may " \
                          "not pose an immediate threat to your health, leaving it untreated could lead " \
                          "to further complications down the road. It is always better to address health " \
                          "concerns as soon as possible, even if they are minor, to ensure that they do " \
                          "not develop into more serious conditions. So, although it may not require urgent " \
                          "attention, it is still recommended to take proactive steps and seek medical advice " \
                          "to prevent the condition from worsening."
                Duration = duration


            # Pass the predicted label to the template for rendering
            contexts = {'Label': predicted_label, 'Message': msg, 'image': img, 'Duration': Duration,
                        'Probability': f'{Proba:.2f}%'}

        elif 'predict-image' in request.POST:
            # Get the uploaded image from the request
            uploaded_file = request.FILES['image']
            # Save the image to a temporary location
            fs = FileSystemStorage()
            file_path = fs.save(uploaded_file.name, uploaded_file)
            # Load the saved image using Keras
            img = image.load_img(file_path, target_size=(224, 224))
            # Preprocess the image for the model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.
            # Use the model to predict the class of the image
            prediction = model.predict(img_array)
            # Get the predicted class label
            predicted_class = np.argmax(prediction[0])
            # Map the predicted class index to a class label
            namesDiseases = ["HairLoss", "Acne and Rosacea", "Actinic Keratosis Basal Cell Carcinoma",
                             "Melanoma Skin Cancer Nevi and Moles", "Vasculties"]

            # Create a dictionary to map disease labels to categories and durations
            disease_info = {
                "hairloss": {
                    "category": "Normal",
                    "duration": "The average duration of treatment is 6 months to 1 year."
                },
                "acne and rosacea": {
                    "category": "Normal",
                    "duration": "The average duration of treatment is 5 to 6 months."
                },
                "actinic keratosis basal cell carcinoma": {
                    "category": "Dangerous",
                    "duration": "The average duration of treatment is 3 months."
                },
                "melanoma skin cancer nevi and moles": {
                    "category": "Dangerous",
                    "duration": "The average duration of treatment is 6 to 12 months."
                },
                "vasculties": {
                    "category": "Dangerous",
                    "duration": "The average duration of treatment is several months to a few years."
                }
            }

            predicted_label = namesDiseases[predicted_class]

            Proba = prediction[0][predicted_class] * 100

            search_str = predicted_label.lower()
            if search_str in disease_info:
                category = disease_info[search_str]["category"]
                duration = disease_info[search_str]["duration"]

                if category == "Dangerous":
                    msg = "It's a DANGEROUS DISEASE, you need to see a doctor. " \
                          "It's important to recognize that some diseases can be very dangerous " \
                          "and require immediate attention from a medical professional. Delaying " \
                          "a doctor's visit for a serious illness can worsen the condition and " \
                          "potentially lead to life-threatening consequences. It's always better " \
                          "to err on the side of caution and seek medical advice as soon as possible, " \
                          "especially if you notice any unusual symptoms or changes in your health. " \
                          "Remember, your health is your most valuable asset, and taking care of " \
                          "it should be a top priority."
                elif category == "Normal":
                    msg = "It is NOT A DANGEROUS DISEASE, you can take your time. " \
                          "If you have been diagnosed with a disease that is not considered serious, " \
                          "it is important to still take it seriously and not ignore it. While it may " \
                          "not pose an immediate threat to your health, leaving it untreated could lead " \
                          "to further complications down the road. It is always better to address health " \
                          "concerns as soon as possible, even if they are minor, to ensure that they do " \
                          "not develop into more serious conditions. So, although it may not require urgent " \
                          "attention, it is still recommended to take proactive steps and seek medical advice " \
                          "to prevent the condition from worsening."
                Duration = duration


            # Pass the predicted label to the template for rendering
            contexts = {'Label': predicted_label, 'Message': msg, 'image': fs.url(file_path), 'Duration': Duration,
                        'Probability': f'{Proba:.2f}%'}

        return render(request, 'prediction.html', contexts)
    return render(request, 'home.html')

def lung(request):
    contexts = {}
    msg = ''
    Duration = ''
    if request.method == 'POST':
        # Get the uploaded image from the request
        uploaded_file = request.FILES['image']
        # Save the image to a temporary location
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        # Load the saved image using Keras
        img = image.load_img(file_path, target_size=(224, 224))
        # Preprocess the image for the model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        # Use the model to predict the class of the image
        prediction = model.predict(img_array)
        # Get the predicted class label
        predicted_class = np.argmax(prediction[0])
        # Map the predicted class index to a class label
        namesDiseases = ["Lung ACA", "Lung N", "Lung SCC"]

        # Create a dictionary to map disease labels to categories and durations
        disease_info = {
            "lung aca": {
                "category": "Dangerous",
                "duration": "Treatment may continue for many months or even years."
            },
            "lung n": {
                "category": "Dangerous",
                "duration": "Treatment may continue for many months or even years"
            },
            "lung scc": {
                "category": "Dangerous",
                "duration": "Treatment may continue for many months or even years"
            }
        }

        predicted_label = namesDiseases[predicted_class]

        Proba = prediction[0][predicted_class] * 100

        search_str = predicted_label.lower()
        if search_str in disease_info:
            category = disease_info[search_str]["category"]
            duration = disease_info[search_str]["duration"]

            if category == "Dangerous":
                msg = "Prioritize regular check-ups and lung health screenings. " \
                      "Avoid smoking and secondhand smoke exposure at all costs. Maintain a healthy lifestyle with a balanced diet and regular exercise " \
                      " If you have any concerns or symptoms, consult a healthcare professional promptly "
            Duration = duration

        # Pass the predicted label to the template for rendering
        contexts = {'Label': predicted_label, 'Message': msg, 'image': fs.url(file_path), 'Duration': Duration,
                    'Probability': f'{Proba:.2f}%'}

        return render(request, 'PredictionLung.html', contexts)
    return render(request, 'lung.html')

def Load(request):
    return render(request, 'LoadingPrediction.html')

def Choice(request):
    return render(request, 'Choix2.html')

def contact(request):
    return render(request, 'contact.html')

def index(request):
    return render(request, 'index.html')
def ChoiceNew(request):
    return render(request, 'NewChoisingPage.html')

def SkinDisease(request):
    return render(request, 'SkinDiseasePage.html')

def blog(request):
    return render(request, 'blog.html')
def LungCancerPage(request):
    return render(request, 'LungCancerPage.html')

def Reviews(request):
    return render(request, 'Reviews.html')
def about(request):
    return render(request, 'about.html')
def team(request):
    return render(request, 'team.html')

def signpro(request):
    return render(request, 'Signature.html')




def prediction(request):
    context = {}
    if request.method == 'POST' and 'image' in request.FILES:
        # Get the uploaded image from the request
        uploaded_file = request.FILES['image']
        # Save the image to a temporary location
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        # Load the saved image using Keras
        img = image.load_img(file_path, target_size=(224, 224))
        # Preprocess the image for the model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        # Use the model to predict the class of the image
        prediction = model.predict(img_array)
        # Get the predicted class label
        predicted_class = np.argmax(prediction[0])
        # Map the predicted class index to a class label
        namesDiseases = ["HairLoss", "Acne and Rosacea", "Actinic Keratosis Basal Cell Carcinoma",
                         "Melanoma Skin Cancer Nevi and Moles", "Vasculties"]

        Normal = ["hairloss", "acne and rosacea", "melanoma skin cancer nevi and moles"]
        Dangerous = ["actinic keratosis basal cell carcinoma", "vasculties"]

        predicted_label = namesDiseases[predicted_class]

        search_str = predicted_label.lower()
        if search_str in Dangerous:
            msg = " It's a dangerous disease , You need to see a doctor"
        elif search_str in Normal:
            msg = "It is not a serious disease, you can take your time"

        # Pass the predicted label to the template for rendering
        context = {'Label': predicted_label, 'Message': msg}
        # context['predicted_label'] = predicted_label
    return render(request, 'prediction.html', context)













from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.mail import EmailMessage, send_mail

from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode


from django.contrib.auth import authenticate, login, logout
from . tokens import generate_token

# Create your views here.
def home(request):
    return render(request, "authentication/index.html")

def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
        
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username.")
            return redirect('home')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('home')
        
        if len(username)>20:
            messages.error(request, "Username must be under 20 charcters!!")
            return redirect('home')
        
        if pass1 != pass2:
            messages.error(request, "Passwords didn't matched!!")
            return redirect('home')
        
        if not username.isalnum():
            messages.error(request, "Username must be Alpha-Numeric!!")
            return redirect('home')
        
        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        # myuser.is_active = False
        myuser.is_active = False
        myuser.save()
      
        
        # Welcome Email
        subject = "Welcome to BANANA king- Django Login!!"
        message = "Hello " + myuser.first_name + "!! \n" + "Welcome to Banana king!! \nThank you for visiting our website\n. We have also sent you a confirmation email, please confirm your email address. \n\nThanking You\nGOSTAVO"        
        from_email = settings.EMAIL_HOST_USER
        to_list = [myuser.email]
        send_mail(subject, message, from_email, to_list, fail_silently=True)
        
        # Email Address Confirmation Email
        current_site = get_current_site(request)
        email_subject = "Confirm your Email @ GFG - Django Login!!"
        message2 = render_to_string('email_confirmation.html',{
            
            'name': myuser.first_name,
            'domain': current_site.domain,
            'uid': urlsafe_base64_encode(bytes(myuser.pk)),
            'token': generate_token.make_token(myuser)
        })
        email = EmailMessage(
        email_subject,
        message2,
        settings.EMAIL_HOST_USER,
        [myuser.email],
        )
        email.fail_silently = True
        email.send()
        
        return redirect('/App/signin/')
        
        
    return render(request, "authentication/signup.html")


def activate(request,uidb64,token):
    try:
        uid = str(urlsafe_base64_decode(uidb64))
        myuser = User.objects.get(pk=uid)
    except (TypeError,ValueError,OverflowError,User.DoesNotExist):
        myuser = None

    if myuser is not None and generate_token.check_token(myuser,token):
        myuser.is_active = True
        # user.profile.signup_confirmation = True
        myuser.save()
        login(request,myuser)
        messages.success(request, "Your Account has been activated!!")
        return redirect('/App/signin/')
    else:
        return render(request,'activation_failed.html')


def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        
        user = authenticate(username=username, password=pass1)
        
        if user is not None:
            login(request, user)
            fname = user.first_name
            # messages.success(request, "Logged In Sucessfully!!")
            return render(request, "authentication/signin.html",{"fname":fname})
        else:
            #messages.error(request, "Bad Credentials!!")
            return redirect('/App/home/')

    return render(request, "authentication/signin.html")


def signout(request):
    logout(request)

    return redirect('/App/home/')