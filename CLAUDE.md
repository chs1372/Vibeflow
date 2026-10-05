# CLAUDE.md — Vibeflow 작업 지침

Vibeflow는 비정렬 격자용 3차원 유한체적 Navier–Stokes 솔버다(C++20, Kokkos,
MPI, PETSc/hypre, CGNS). 새 기법은 Python 참조 구현(`prototype/`)으로 먼저 쓰고
검증한 뒤 C++로 옮기며, Python 출력이 C++ 시험의 기준값이 된다. 이 저장소에서
믿을 근거는 코드보다 먼저 쓰인 검증 게이트뿐이다.

현황은 세 곳에서 본다. README.md의 Status와 Known limits, ROADMAP.md의 마지막
차수와 마지막 '다음 작업', 그리고 docs/DECISIONS.md의 마지막 ADR이다.

## 작업 규칙

이 규칙들은 상황에 따라 조정하지 않는다. 지키기 어려운 상황이 오면 우회하지 말고
사용자에게 묻는다.

1. **게이트와 판정 기준이 먼저다.** 각 단계의 검증 게이트와 판정 기준은 코드와
   실행보다 먼저 ADR로 써서 커밋한다. ADR에는 쓰기 전에 이미 본 데이터를 적는다
   (ADR-040·042의 "Data seen before this was written", ADR-043의 "Looked at by
   then"). 가장 최근의 순서는 ADR-045(21차)다.
   `docs: ADR-045, ... stated before the code and the runs` → 하네스와 게이트
   스크립트 → 실행과 개정 → `docs: ADR-045 answered -- ...` →
   `README, ROADMAP (round 21, doc rev 71): ...`. 새 기법(v2a, ADR-041, ADR-042)은
   Python 게이트를 먼저, 이어서 C++ 게이트를 스텁에 대해 실패하는 상태로
   커밋했다. 이때 ADR 본문이 실패하는 게이트와 같은 커밋에 들어가기도 했다.
   벤치마크나 원인 조사(ADR-043–045)에는 그 단계가 없다.
2. **게이트는 절대 느슨하게 하지 않는다.** 실패는 실패로 판정하고 그대로
   기록한다(ADR-043, ADR-045). 기준이나 측정 방법이 잘못됐다는 것을 알게 되면,
   판정할 결과를 보기 전에 ADR 개정으로 고치고 그때까지 본 것을 함께 적는다.
   앞선 실패도 옆에 그대로 보고한다(ADR-043의 두 번째·세 번째 시험). 이미 본
   결과를 통과시키려고 기준을 바꾸지 않는다.
3. **결과 전후의 모든 수정은 ADR에 공개한다.** 결과 전의 수정은 그 ADR 끝에
   기울임 단락으로 덧붙이고, 그때까지 본 것을 함께 적는다. ADR-045는 여덟 번
   그렇게 했다(`*Revision before the runs.*`, `*Second revision, before the
   results.*`, …). 게이트가 예상하지 못한 수정은 결과 단락에서 밝힌다(ADR-044의
   BoomerAMG 최하층 이완, "which the gate did not foresee and is disclosed
   here"). 솔버 코드를 바꿔야 하면 바꾸기 전에 ADR에 적고, 바꾼 뒤에는 전체
   스위트(`run_gates.py v0 v1 v2 v2b`)를 돌린다. 이것은 ADR-045가 자기 사례에
   정한 규칙이고, ADR-041과 ADR-044도 그렇게 했다.
4. **docs/DECISIONS.md는 덧붙이기만 한다.** 이미 쓴 문장은 고치거나 지우지
   않는다. 새 ADR은 파일 끝에 붙이고, 진행 중인 ADR의 개정과 결과는 그 항목의
   끝에 붙인다. 뒤에 다른 ADR이 이미 있어도 그렇게 한다(ADR-043, ADR-044).
   최근 ADR은 처음부터 제목에 "Stated before …, answered after"를 넣으므로 나중에
   제목을 고칠 일이 없다. 예전 항목이 틀렸으면 새 ADR에서 바로잡는다. 원문은
   남기고 예전 항목에는 새 ADR을 가리키는 짧은 표지만 넣는다(`**[Corrected by
   ADR-027]**`, `*Correction (ADR-040):*`, `**Closed by ADR-026**`). v1 시기에는
   ADR-012·016·022·024·026을, 그 뒤에는 ADR-037을 한 번 제자리에서 고쳐 쓴 일이
   있었다. 그 방식은 이제 쓰지 않는다.
5. **ROADMAP.md의 차수 기록은 한국어로 쓴다.** 최근 차수(19–21차)의 형식을
   따른다. `### N차 — 제목`, 소개 단락, 굵은 꼬리표가 붙은 항목들, 그리고
   `**다음:**` 한 줄이다. 같은 커밋에서 맨 아래의 '현재 게이트 상태' 표(게이트가
   바뀌었을 때)와 '다음 작업' 체크리스트를 갱신한다. 그보다 위에 있는 옛 표와
   목록은 그대로 둔다. ROADMAP.md는 Claude 문서 「CFD 솔버 아키텍처 및 개발
   로드맵」을 내보낸 사본이고, 원본은 그 문서다. 원본에 접근할 수 없으면
   ROADMAP.md만 고치되 맨 아랫줄의 동기화 날짜와 rev는 바꾸지 않는다. 그리고
   사용자에게 알려 원본을 맞추게 한다. README, DECISIONS, 코드 주석, 커밋
   메시지는 영어로 쓴다(ROADMAP.md가 유일한 한국어 문서다).
6. **커밋 작성자는 chs1372다.** 모든 커밋의 작성자가
   `chs1372 <84263517+chs1372@users.noreply.github.com>`이다. git 설정은 바꾸지
   않는다. Windows 쪽 git에는 user.name과 user.email이 없으므로 커밋할 때마다
   `-c`로 넘긴다.

   ```sh
   git -c user.name=chs1372 -c user.email=84263517+chs1372@users.noreply.github.com commit
   ```

   메시지는 영어로 쓴다. 제목은 대개 `영역: 무엇` 꼴이다(`docs:`, `tests:`,
   `prototype:`, `physics:`, `backstep:` 등). 끝에는 하네스가 정한 귀속 줄을
   붙인다(`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
7. **단계 순서.** run_gates.py 머리말은 "A stage's gates must pass before code
   for the next stage is merged"라고 정한다. ROADMAP의 리스크 표는 "게이트 미통과
   시 새 물리 추가 금지를 규칙으로"라고 적는다. '다음 작업'에는 "v2c: 벽함수,
   복합 열전달 — v2b 통과 후 게이트 명시"가 있다. v2b는 README에서 'in progress'다.
   후향계단의 실패한 게이트는 스위트 밖에서 돌기 때문에, 스위트가 통과했다고
   v2b가 통과한 것은 아니다. ROADMAP 21차 '다음'은 두 길을 적는다. 하나는 "계단
   끝 흡입의 원인을 따로 ADR로 먼저 정한다", 다른 하나는 "한계로 두고 v2c로
   넘어간다"이다. 어느 쪽으로 갈지는 사용자에게 묻는다.
8. **원인 조사도 같은 방식으로 한다.** 가설, 측정, 그리고 '원인을 찾았다'고 말할
   규칙을 먼저 ADR로 커밋한다(원인 조사: ADR-029–032, ADR-043; 사전 등록한
   시험 전반: ADR-033, 034, 036, 040). 규칙이 원인을 확정하지 못하면 원인으로
   부르지 않는다. 본 것만 기록하고 Known limits로 보낸다(ADR-043).
9. **아직 보지 않은 데이터는 ADR보다 먼저 보지 않는다.** 지금 이 규칙에 묶인 것은
   TMR의 FUN3D 후향계단 결과와 TMR의 가장 고운 후향계단 격자다. 둘 다 저장소에
   없다. ADR-045의 결정이 이 둘을 "stated in an ADR of its own before it is
   looked at"으로 묶었다. 웹에서 받거나 열기 전에 ADR-046을 먼저 커밋한다.

## 저장소 구조

```
src/core            Kokkos 타입, MPI 통신기
src/mesh            기하, 생성 격자, .hex·.vmesh·CGNS 읽기, 분할기
src/linalg          선형계, 네이티브 CG·BiCGStab, PETSc/hypre 백엔드
src/discretization  기울기, 대류·확산 연산자, 면 플럭스
src/physics         PISO/PIMPLE: 운동량, Rhie–Chow, 압력, BDF2, 에너지, 부력, SST, 벽거리
src/io              VTK XML 출력(.vtu/.pvtu)
prototype/          모든 기법의 Python 참조 구현, dump_fixtures.py
tests/mms           MMS·교차검증·MPI 게이트와 게이트 실행기 run_gates.py
tests/unit          단위 시험(기하, CGNS, VTU, 백엔드), check_vtu.py
tests/fixtures      Python 기준값 픽스처와 CGNS 픽스처
tests/benchmark     cavity, cylinder, checkerboard.py, heated_cavity, rayleigh_benard,
                    rb_growth·rb_linear.py, flat_plate(+_gate.py), backstep(+_gate.py,
                    _report.py), ADR-043의 모드 모형 piso_mode.py·flat_plate_mode.py
cases/              cylinder(gmsh 스크립트), flatplate·backstep(NASA TMR 격자와 기준값,
                    backstep은 실험값도)
tools/              C++ 픽스처 작성기(CGNS 픽스처 포함), 격자 품질 보고
docs/               DECISIONS.md(ADR), LICENSING.md
```

CMake 타깃 의존성이 계층을 강제한다. 한 계층은 자기보다 아래 계층만 링크한다.
위로 향하는 링크는 빌드 수정이 아니라 설계 변경이다(CMakeLists.txt). GPL 도구는
`src/`에서 링크하지 않는다. gmsh는 `cases/cylinder/make_mesh.py`가 Python API로만
쓰고, 솔버는 그 스크립트가 쓴 `.hex`만 읽는다(docs/LICENSING.md).

게이트는 환경 변수를 하나도 설정하지 않은 채로 판정한다. 하네스가
"Exploration only"라고 표시한 변수(예: backstep.cpp 머리말)는 판정 실행에 쓰지
않는다. 솔버의 진단 스위치(`TurbulenceModel.frozen`, `transposeStress`, `bdf1`,
`momentumSolveTol`, `rhieChowAxisOff`)는 기본값이 꺼진 채로 트리에 남긴다
(ADR-042, ADR-043).

픽스처를 바꾸는 Python 변경 뒤에는 `python prototype/dump_fixtures.py`를
실행하고 `git diff tests/fixtures/`를 확인한다. CI(ubuntu-24.04)는 이 픽스처
검사와 v0만 돌린다. v1부터의 단계와 후향계단은 로컬에서만 돌고, 결과는
DECISIONS.md에 기록된다. 초록색 CI가 그 단계들까지 덮는다고 읽지 않는다.

## 빌드와 실행(이 PC)

**환경.** WSL2 Ubuntu 26.04.1, i5-9600K 6코어(SMT 없음), RAM 31 GB. GCC 15.2,
CMake 4.2.3, Python 3.14이다. README의 apt 목록을 이 배포판에서 설치하면 OpenMPI
5.0.10, PETSc 3.24.4, hypre 3.0.0, CGNS 4.5.0이 들어오고, Kokkos 5.2.2는 소스로
빌드한다. README의 검증 구성(Ubuntu 24.04, GCC 13.3, OpenMPI 4.1, PETSc 3.19,
hypre 2.28, CGNS 3.4)이나 CI와 다르다.

**기록된 숫자는 두 코어 기계에서 나왔다.** 후향계단은 두 스레드에서 돌았다.
run_gates.py는 직렬 게이트의 `OMP_NUM_THREADS`를 정하지 않으므로, 이 PC에서는
Kokkos가 여섯 코어를 다 쓴다. 스레드 수가 바뀌면 리덕션 순서가 바뀌어 마지막
자리가 달라진다(ADR-034, ADR-040). 기록과 비교할 때는 `OMP_NUM_THREADS=2`와
`VIBEFLOW_BS_THREADS=2`로 돌린다. 마지막 자리가 어긋나면 툴체인과 스레드 수를
먼저 의심하고, 그 차이도 함께 보고한다.

**빌드와 실행은 WSL 안의 클론 `~/Vibeflow`에서 한다.** Windows 쪽
체크아웃(이 세션의 작업 디렉터리)은 편집과 커밋에만 쓴다. 이유는 네 가지다.
Windows 체크아웃은 `core.autocrlf=true` 때문에 CRLF라서 sh 스크립트가 깨진다.
C: 드라이브 여유가 8 GB 남짓이다. `/mnt/c`는 느리다. 그리고 워크트리의 `.git`이
Windows 경로를 가리켜 WSL의 git이 읽지 못한다.

- `~/Vibeflow`의 origin은 Windows 저장소
  `/mnt/c/Users/CHS/Desktop/프로젝트/CFD/Vibeflow`다. Windows 쪽에서 커밋한 뒤
  WSL로 가져온다. `~/Vibeflow`에서는 커밋하지 않는다.

  ```sh
  cd ~/Vibeflow && git fetch origin && git checkout -B <branch> origin/<branch>
  ```

- Windows의 Git Bash에서 WSL 명령을 부를 때는 경로 변환을 막아야 한다:
  `MSYS_NO_PATHCONV=1 wsl.exe --cd /home/chs -e bash -lc '...'`

**의존성**은 README의 'Build'와 같다. sudo에는 비밀번호가 필요하므로 apt 설치는
사용자가 직접 실행한다. README의 목록에 gmsh가 헤드리스 환경에서 쓰는 라이브러리를
더하면 다음과 같다.

```sh
sudo apt-get install -y build-essential cmake git pkg-config python3-venv \
    libopenmpi-dev openmpi-bin libcgns-dev libpetsc-real-dev libhypre-dev \
    libgl1 libglu1-mesa libxcursor1 libxft2 libxinerama1
```

Kokkos는 `~/kokkos-install`에, venv는 `~/Vibeflow/.venv`에 있다(numpy, scipy,
meshio, h5py, sympy, gmsh 4.15.2). 빌드 디렉터리는 반드시 저장소 루트의
`build`여야 한다. run_gates.py가 `build/tests`를 하드코딩하고 있어서, 다른
디렉터리에 빌드하면 C++ 게이트가 모두 'not built'로 건너뛰어진다.

```sh
cd ~/Vibeflow && . .venv/bin/activate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT=$HOME/kokkos-install
cmake --build build -j6
```

**게이트.** 반드시 venv를 켠 채로 돌린다. Python 게이트와 격자 스크립트가
`sys.executable`(`python3`)로 실행되기 때문이다. 단계 키별 게이트 수는 v0 8개,
v1 15개, v2 10개, v2b 6개다. v2a의 키는 `v2`다. 모르는 키는 아무것도 돌리지
않고도 "All active gates passed"를 출력한다. 건너뛴 게이트(SKIPPED)도 마찬가지로
그 문구를 막지 않는다. SKIPPED 줄을 읽고, 건너뛴 게이트를 통과로 세지 않는다.

```sh
python3 tests/mms/run_gates.py v0     # 약 1분
sh cases/cylinder/make_meshes.sh      # v1 전에: 원통 격자 두 개
python3 tests/mms/run_gates.py v1     # 두 코어에서 약 1시간
python3 tests/mms/run_gates.py v2     # 몇 시간(가열 공동만 약 2.5시간)
python3 tests/mms/run_gates.py v2b    # 평판만 여러 시간
python3 tests/benchmark/backstep_gate.py build build/backstep   # 스위트 밖, 두 스레드로 반나절
python3 tests/benchmark/backstep_report.py build/backstep
```

평판 격자는 flat_plate_gate.py가, 후향계단 level 1·2 격자는 backstep_gate.py가
없으면 직접 만든다. `VIBEFLOW_FP_REUSE=1`은 이미 돈 평판 실행을 판정만 한다.
`VIBEFLOW_SUITE_SKIP=<조각,...>`은 요청한 모든 단계에서 이름에 그 조각이 들어간
게이트를 건너뛴다. 대소문자를 구분하는 부분 문자열 비교다. "mpi" 같은 넓은
조각 대신 "de Vahl Davis"나 "NASA TMR" 같은 고유한 조각을 쓴다.

**긴 실행.** 여러 시간짜리 실행은 백그라운드로 돌리고 로그를 남긴다. 후향계단
행진은 500스텝마다 체크포인트를 남긴다. 중단된 뒤에는 반드시
`VIBEFLOW_BS_RESUME=1`로 이어 간다. 그 변수 없이 backstep_gate.py를 다시
돌리면 체크포인트가 지워지고 로그를 덮어쓴다. `VIBEFLOW_BS_REUSE=1`은 이미
끝난(FINAL 줄이 있는) 행진을 건너뛴다. 스레드 수는 `VIBEFLOW_BS_THREADS`로
정하며 기본값은 1이다. 출력 디렉터리의 `threads` 파일이 있으면 그것이 우선하고,
다음 행진부터 적용된다. 게이트 실행기는 `OMP_PROC_BIND=false`를 쓴다. MPI
게이트는 `mpirun --oversubscribe --allow-run-as-root -n N`에 `OMP_NUM_THREADS=1`로
돈다.
