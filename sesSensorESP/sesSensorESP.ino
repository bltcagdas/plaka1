#include <ESP8266WiFi.h>
#include <WiFiClientSecure.h>
#include <ESP8266HTTPClient.h>

// 🔧 İlk sefer için SSID/PASS yaz (sonra flash'ta kalır)
const char* WIFI_SSID = "GALAXY NOTE 20 ULTRA";
const char* WIFI_PASS = "aabbccdd";

// ✅ Firebase RTDB
const char* FIREBASE_HOST = "try1-cc8eb-default-rtdb.europe-west1.firebasedatabase.app";
const char* STATUS_PATH   = "/test/status.json";

// ✅ Analog / threshold
const int THRESHOLD = 1000;

// ✅ Spam engelleme (Firebase yazma)
unsigned long lastSendMs = 0;
const unsigned long COOLDOWN_MS = 2000;
bool lastSentYes = false;

// ✅ Wi-Fi reconnect kontrolü
unsigned long lastWifiAttemptMs = 0;
const unsigned long WIFI_RETRY_MS = 5000; // 5 sn'de bir dene

bool firebaseSetStatus(const String& value) {
  if (WiFi.status() != WL_CONNECTED) return false;

  WiFiClientSecure client;
  client.setInsecure(); // hızlı prototip için

  HTTPClient https;
  String url = String("https://") + FIREBASE_HOST + STATUS_PATH;

  if (!https.begin(client, url)) {
    Serial.println("❌ HTTPS begin hatasi");
    return false;
  }

  https.addHeader("Content-Type", "application/json");

  int httpCode = https.PUT("\"" + value + "\"");
  String payload = https.getString();
  https.end();

  Serial.print("Firebase PUT code: ");
  Serial.print(httpCode);
  Serial.print(" | resp: ");
  Serial.println(payload);

  return (httpCode > 0 && httpCode < 400);
}

void wifiInitAndAutoConnect() {
  WiFi.mode(WIFI_STA);

  // ✅ Flash'a yaz (kalıcı)
  WiFi.persistent(true);
  WiFi.setAutoConnect(true);
  WiFi.setAutoReconnect(true);

  // Eğer daha önce kayıtlı ağ varsa: WiFi.begin() parametresiz de deneyebilir
  // ama ilk kurulum için biz SSID/PASS verip flash'a yazdırıyoruz:
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi: ilk baglanti denemesi (flash'a kaydedilecek)...");
    WiFi.begin(WIFI_SSID, WIFI_PASS);
  }
}

void wifiEnsureConnected() {
  // ✅ Bağlıysa hiçbir şey yapma
  if (WiFi.status() == WL_CONNECTED) return;

  unsigned long now = millis();
  if (now - lastWifiAttemptMs < WIFI_RETRY_MS) return;
  lastWifiAttemptMs = now;

  Serial.println("⚠ WiFi baglantisi yok. Tekrar baglaniliyor...");

  // Burada tekrar begin çağırabiliriz. Kayıtlı SSID/PASS flash'ta.
  // İstersen parametresiz de deneyebilirsin:
  // WiFi.begin();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void setup() {
  Serial.begin(115200);
  delay(200);

  wifiInitAndAutoConnect();

  // İlk bağlantı için kısa bir bekleme
  Serial.print("WiFi baglaniyor");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("✅ Baglandi. IP: ");
    Serial.println(WiFi.localIP());
    firebaseSetStatus("no");
  } else {
    Serial.println("❌ WiFi baglanamadi (15 sn). Loop içinde denemeye devam edecek.");
  }
}

void loop() {
  // ✅ Bağlı değilse aralıklı reconnect (bağlıysa hiç denemez)
  wifiEnsureConnected();

  // WiFi yoksa Firebase'e gitme
  if (WiFi.status() != WL_CONNECTED) {
    delay(200);
    return;
  }

  int sensorValue = analogRead(A0);
  Serial.print("Analog(A0): ");
  Serial.println(sensorValue);

  unsigned long now = millis();

  if (sensorValue >= THRESHOLD) {
    if (!lastSentYes && (now - lastSendMs > COOLDOWN_MS)) {
      Serial.println("🔊 THRESHOLD geçti -> status YES");
      if (firebaseSetStatus("yes")) {
        lastSentYes = true;
        lastSendMs = now;
      }
    }
  } else {
    if (lastSentYes && (now - lastSendMs > 500)) {
      Serial.println("🔇 Altina dustu -> status NO");
      if (firebaseSetStatus("no")) {
        lastSentYes = false;
        lastSendMs = now;
      }
    }
  }

  delay(100);
}
