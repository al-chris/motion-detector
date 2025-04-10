#include <WiFi.h>
#include <WebSocketClient.h>
#include <esp32cam.h>

#include "esp_camera.h"


#include <mbedtls/base64.h>  // Use the built-in mbedtls base64 implementation

const int LED_PIN = 4;
const char *host = "192.168.137.1";
char wspath[] = "/ws/video-stream";
char wshost[] = "192.168.137.1:8000";

WebSocketClient webSocketClient;
WiFiClient client;

void setup() {
  Serial.begin(115200);
  delay(10);
  WiFi.mode(WIFI_STA);
  
  WiFi.begin("DESKTOP-J7H5805 8266", "s99(630L");

  // Initialize camera
  pinMode(LED_PIN, OUTPUT);

  auto res = esp32cam::Resolution::find(320, 240); // Lower resolution for easier transmission
  esp32cam::Config cfg;
  cfg.setPins(esp32cam::pins::AiThinker);
  cfg.setResolution(res);
  cfg.setJpeg(10);
  cfg.setBufferCount(1);
  
  bool ok = esp32cam::Camera.begin(cfg);
  if (!ok) {
    Serial.println("Camera init failed");
    delay(3000);
    ESP.restart(); // Restart the ESP32 if camera init fails
  }

  delay(3000);

  // Setup status LED
  pinMode(4, OUTPUT);
  digitalWrite(4, LOW);

  delay(1000);
  Serial.println("Connecting Wifi...");
  
  // Wait for WiFi connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  connectToServer();
}

bool connectToServer() {
  // Connect to the server
  Serial.println("Connecting to server...");
  if (client.connect(host, 8000)) {
    Serial.println("Connected");
    
    // Handshake with the server
    webSocketClient.path = wspath;
    webSocketClient.host = wshost;
    
    Serial.println("Initiating handshake...");
    if (webSocketClient.handshake(client)) {
      Serial.println("Handshake successful");
      return true;
    } else {
      Serial.println("Handshake failed");
      return false;
    }
  } else {
    Serial.println("Connection failed");
    return false;
  }
}

// Helper function for base64 encoding
String base64Encode(const uint8_t* data, size_t length) {
  size_t outputLength = 0;
  mbedtls_base64_encode(NULL, 0, &outputLength, data, length);  // Get required buffer size
  
  size_t bufferSize = outputLength + 1;  // +1 for null terminator
  char* buffer = (char*)malloc(bufferSize);
  if (!buffer) {
    Serial.println("Memory allocation failed");
    return "";
  }
  
  mbedtls_base64_encode((unsigned char*)buffer, bufferSize, &outputLength, data, length);
  buffer[outputLength] = 0;  // Ensure null termination
  
  String result = String(buffer);
  free(buffer);
  return result;
}

void handleStream() {
  Serial.println("Starting Stream.");
  digitalWrite(LED_PIN, HIGH); // Use GPIO4
  
  if (!client.connected()) {
    Serial.println("Client not connected");
    digitalWrite(LED_PIN, LOW);
    return;
  }
  
  auto startTime = millis();
  int frameCount = 0;
  int consecutiveFailures = 0;
  
  while (client.connected()) {
    // Log available memory
    Serial.printf("Free heap: %d\n", ESP.getFreeHeap());

    // Explicitly free memory before capture
    esp_camera_fb_return(nullptr);

    // Capture a frame from the camera
    auto img = esp32cam::capture();
    if (img == nullptr) {
      Serial.println("Camera capture failed");
      delay(500);
      consecutiveFailures++;
      
      if (consecutiveFailures > 5) {
        Serial.println("Too many consecutive failures, restarting camera");
        esp32cam::Camera.end();
        delay(1000);
        
        auto res = esp32cam::Resolution::find(320, 240);
        esp32cam::Config cfg;
        cfg.setPins(esp32cam::pins::AiThinker);
        cfg.setResolution(res);
        cfg.setJpeg(10);
        cfg.setBufferCount(1);
        
        bool ok = esp32cam::Camera.begin(cfg);
        if (!ok) {
          Serial.println("Camera reinit failed, restarting device");
          ESP.restart();
        }
        consecutiveFailures = 0;
      }
      continue;
    }
    
    consecutiveFailures = 0; // Reset counter on successful capture
    
    // Base64 encode the image data
    String base64Image = base64Encode(img->data(), img->size());
    if (base64Image.length() == 0) {
      Serial.println("Base64 encoding failed");
      img = nullptr;
      delay(1000);
      continue;
    }
    
    // Get timestamp
    String timestamp = "2025-04-08T15:25:53Z"; // Current timestamp
    
    // Create the complete JSON message
    String jsonMessage = "{\"frame\":\"data:image/jpeg;base64," + base64Image + 
                  "\",\"timestamp\":\"" + timestamp + "\"}";
    
    // Send the complete JSON message
    Serial.println("Sending frame...");
    webSocketClient.sendData(jsonMessage);
    
    // Free resources
    img = nullptr;
    
    frameCount++;
    
    // Check for any incoming messages from server
    String response;
    if (webSocketClient.getData(response)) {
      if (response.length() > 0) {
        Serial.println("Server response: " + response);
      }
    }
    
    delay(200); // Control frame rate
  }
  
  auto duration = millis() - startTime;
  Serial.printf("STREAM END %dfrm %0.2ffps\n", frameCount, 1000.0 * frameCount / duration);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  if (client.connected()) {
    // Handle the video stream
    handleStream();
  } else {
    Serial.println("Client disconnected.");
    digitalWrite(4, LOW);
    
    // Try to reconnect
    int retries = 0;
    while (!connectToServer() && retries < 5) {
      delay(1000);
      retries++;
    }
    
    if (retries >= 5) {
      // Reset ESP32 if unable to connect after 5 attempts
      Serial.println("Failed to reconnect after 5 attempts. Restarting...");
      ESP.restart();
    }
  }
}