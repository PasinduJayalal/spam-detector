import joblib

def main():
    
    sms_data = []
    email_data = []

    while True:
        try:
            model_choice = input("Choose model to use (sms/email) or 'exit' to quit: ").strip().lower()
            if model_choice == 'exit':
                break
            if model_choice not in ['sms', 'email']:
                print("Invalid choice. Please choose 'sms' or 'email'.")
                continue
            if model_choice == 'sms':
                sms_data.append(input("Enter SMS text: "))
            elif model_choice == 'email':
                email_data.append(input("Enter Email text: "))
        except EOFError:
            break
    # while True:
    #     try:
    #         sms_text = input("Enter SMS text (or 'exit' to quit): ")
    #         if sms_text.lower() == 'exit':
    #             break
    #         sms_data.append(sms_text)
    #     except EOFError:
    #         break
        
    # while True:
    #     try:
    #         email_text = input("Enter Email text (or 'exit' to quit): ")
    #         if email_text.lower() == 'exit':
    #             break
    #         email_data.append(email_text)
    #     except EOFError:
    #         break

    sms = joblib.load('models/sms_pipeline.pkl')
    email = joblib.load('models/email_pipeline.pkl')

    for sms_text in sms_data:
        print(f"SMS: {sms_text}")
        sms_pred = sms.predict([sms_text])
        print(f"Predicted: {'Spam' if sms_pred[0] == 1 else 'Not Spam'}")

    for email_text in email_data:
        print(f"Email: {email_text}")
        email_pred = email.predict([email_text])
        print(f"Predicted: {'Spam' if email_pred[0] == 1 else 'Not Spam'}")
        

if __name__ == "__main__":
    main()
    
    
