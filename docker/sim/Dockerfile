FROM eclipse-temurin:24
RUN mkdir /experiment
WORKDIR /experiment
COPY effects effects
COPY gradle gradle
COPY src src
COPY .java-version ./
COPY docker docker
COPY *.kts ./
COPY *.properties ./
RUN sed -i '/alias(libs.plugins.gitSemVer)/d' build.gradle.kts
RUN sed -i '/alias(libs.plugins.multiJvmTesting)/d' build.gradle.kts
RUN sed -i '/multiJvm {/,/}/d' build.gradle.kts
COPY gradlew* ./
RUN CI=true ./gradlew runAllGraphic
RUN rm -rf data
RUN ./gradlew --stop
CMD ./gradlew runAllBatch
