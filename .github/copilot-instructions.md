# Copilot Instructions for experiment-2025-gradcast

## Project Overview
This is an experimental repository for GradCast using the Alchemist simulator with Collektive for collective programming. The project simulates distributed systems and collective behavior.

## Technology Stack
- **Language**: Kotlin (JVM)
- **Build Tool**: Gradle with Kotlin DSL
- **Framework**: Alchemist simulator
- **Collective Programming**: Collektive framework
- **Java Version**: Supports Java 21 and Java 23 (configured via `.java-version` file)
- **Testing**: Multi-JVM testing with both Java 21 and current version
- **Python**: Used for data processing and chart generation
- **Docker**: Containerized execution environment

## Build System
- Main build file: `build.gradle.kts`
- Java version is read from `.java-version` file
- Uses Gradle toolchains for JVM version management
- Multi-JVM testing support via `multiJvmTesting` plugin
- Supports both Java 21 and Java 23 for compilation and testing
- To switch Java versions: Update `.java-version` file with desired version

## Key Components
- **Simulation Files**: Located in `src/main/yaml/` (YAML configuration files)
- **Effects**: Visual effects configurations in `effects/` directory
- **Data Processing**: Python scripts for generating charts from simulation data
- **Docker**: Separate containers for simulations and chart generation

## Development Workflow
1. **Local Development**: Use `./gradlew` for building and testing
2. **Running Simulations**: 
   - Graphic mode: `./gradlew run<SimulationName>Graphic`
   - Batch mode: `./gradlew run<SimulationName>Batch`
3. **Chart Generation**: Run `python process.py` after simulations
4. **Docker**: Use `docker compose up --build` for containerized execution

## Code Conventions
- Follow Kotlin coding standards
- Use meaningful variable names in simulation configurations
- Organize simulation YAML files by experiment type
- Keep effect configurations modular and reusable

## Testing
- Automated testing via GitHub Actions
- Multi-platform testing (Windows, macOS, Linux)
- Container-based testing for reproducibility
- Short CI runs with 2-second time limits for testing

## Dependencies
- Alchemist framework for simulation engine
- Collektive for collective programming primitives
- Kotlin standard library and reflection
- Platform-specific UI components (SwingUI when not headless)

## Configuration Files
- `.java-version`: Specifies Java version for toolchain
- `gradle.properties`: Gradle configuration
- `renovate.json`: Dependency update automation
- `docker-compose.yml`: Container orchestration

## Performance Considerations
- Heap size auto-detection based on available RAM
- Parallel execution support for batch simulations
- Memory-efficient simulation parameters
- Optimized Docker image layers

## Common Issues
- Ensure Java version compatibility when modifying `.java-version` (Java 21+ required)
- Both Java 21 and Java 23 are supported and tested
- Graphics subsystem requires non-headless environment
- Batch simulations need sufficient heap space
- Docker containers require proper CI environment variables