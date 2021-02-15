import org.jetbrains.kotlin.gradle.tasks.KotlinCompile


plugins {
    kotlin("jvm") version "1.4.10"
    application
}

group = "net.dolpen.nn"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-beta7")
    // for CPU
    implementation("org.nd4j:nd4j-native-platform:1.0.0-beta7")
    // using GPU
    //implementation("org.nd4j:nd4j-cuda-10.0-platform:1.0.0-beta7")
}

tasks.withType<KotlinCompile>(){
    kotlinOptions.jvmTarget = "1.8"
}
application {
    mainClassName = "net.dolpen.nn.MainKt"
}