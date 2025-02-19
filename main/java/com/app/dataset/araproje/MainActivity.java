package com.app.dataset.araproje;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;


import com.app.dataset.araproje.ml.Model97;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


public class MainActivity extends AppCompatActivity {


    private EditText pregnanciesInput, glucoseInput, bloodPressureInput, skinThicknessInput,
            insulinInput, bmiInput, diabetesPedigreeInput, ageInput;
    private TextView resultTextView;
    private Button predictButton;

    // Normalizasyon için gerekli mean ve scale değerleri
    // Normalizasyon için ortalama (mean) ve standart sapma (scale) değerleri
    float[] mean = {3.927f, 126.27667028f, 72.84946948f, 30.15796598f, 167.9555826f,
            33.07928358f, 0.49304175f, 34.005f};
    float[] scale = {3.27225778f, 31.11031505f, 11.60340647f, 8.62219536f, 87.85045441f,
            6.64597032f, 0.33197327f, 11.3247947f};
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // EditText'leri tanımla
        pregnanciesInput = findViewById(R.id.input_pregnancies);
        glucoseInput = findViewById(R.id.input_glucose);
        bloodPressureInput = findViewById(R.id.input_blood_pressure);
        skinThicknessInput = findViewById(R.id.input_skin_thickness);
        insulinInput = findViewById(R.id.input_insulin);
        bmiInput = findViewById(R.id.input_bmi);
        diabetesPedigreeInput = findViewById(R.id.input_diabetes_pedigree);
        ageInput = findViewById(R.id.input_age);
        resultTextView = findViewById(R.id.result_text);
        // Button'ı tanımla
        predictButton = findViewById(R.id.button_predict);
        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try {
                    // Kullanıcı girişlerini al
                    float[] inputs = new float[8];
                    inputs[0] = Float.parseFloat(pregnanciesInput.getText().toString());
                    inputs[1] = Float.parseFloat(glucoseInput.getText().toString());
                    inputs[2] = Float.parseFloat(bloodPressureInput.getText().toString());
                    inputs[3] = Float.parseFloat(skinThicknessInput.getText().toString());
                    inputs[4] = Float.parseFloat(insulinInput.getText().toString());
                    inputs[5] = Float.parseFloat(bmiInput.getText().toString());
                    inputs[6] = Float.parseFloat(diabetesPedigreeInput.getText().toString());
                    inputs[7] = Float.parseFloat(ageInput.getText().toString());


// Normalizasyon işlemi
                    float[] normalizedInputs = new float[8];
                    for (int i = 0; i < inputs.length; i++) {
                        normalizedInputs[i] = (inputs[i] - mean[i]) / scale[i];
                    }

                    Log.d("Input Values", "Pregnancies: " + inputs[0] + ", Glucose: " + inputs[1] +
                            ", Blood Pressure: " + inputs[2] + ", Skin Thickness: " + inputs[3] +
                            ", Insulin: " + inputs[4] + ", BMI: " + inputs[5] +
                            ", Diabetes Pedigree: " + inputs[6] + ", Age: " + inputs[7]);

// Normalizasyonlu veriyi logla
                    Log.d("Normalized Values", "Normalized Pregnancies: " + normalizedInputs[0] +
                            ", Normalized Glucose: " + normalizedInputs[1] + ", Normalized Blood Pressure: " + normalizedInputs[2] +
                            ", Normalized Skin Thickness: " + normalizedInputs[3] + ", Normalized Insulin: " + normalizedInputs[4] +
                            ", Normalized BMI: " + normalizedInputs[5] + ", Normalized Diabetes Pedigree: " + normalizedInputs[6] +
                            ", Normalized Age: " + normalizedInputs[7]);
                    try {
                        // TensorFlow Lite modelini yükleyin
                        Model97 model = Model97.newInstance(MainActivity.this);

                        // Model için inputFeature0 oluşturuluyor
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 8}, DataType.FLOAT32);
                        // Normalleştirilmiş veriyi ByteBuffer formatında model için hazırlıyoruz
                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 8); // 8 float değer
                        byteBuffer.order(ByteOrder.nativeOrder());
                        for (float value : normalizedInputs) {
                            byteBuffer.putFloat(value); // Normalizasyonlu veriyi ByteBuffer'a ekle
                        }
                        Log.d("ByteBuffer Info", "ByteBuffer size: " + byteBuffer.remaining());  // Bu satırı burada ekleyin

                        inputFeature0.loadBuffer(byteBuffer);

                        // Tahmin işlemi
                        Model97.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        // Modelin tahmin sonuçları
                        float result = outputFeature0.getFloatArray()[0];
                        Log.d("Model Prediction", "Model prediction result: " + result);

                        if (result < 0.5) {
                            resultTextView.setText("Tahmin: Diabet");
                        } else {
                            resultTextView.setText("Tahmin: Diabet değil");
                        }

                        // Model kaynaklarını serbest bırak
                        model.close();

                    } catch (IOException e) {
                        Log.e("Model Error", "Model yüklenemedi: " + e.getMessage());
                        Toast.makeText(MainActivity.this, "Model yüklenemedi", Toast.LENGTH_SHORT).show();
                    }

                } catch (NumberFormatException e) {
                    Toast.makeText(MainActivity.this, "Geçersiz giriş verisi", Toast.LENGTH_SHORT).show();
                    return;  // Geri dönerek işlemi durdurun
                }
            }
        });
    }
}