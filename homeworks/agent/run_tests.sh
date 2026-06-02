#!/usr/bin/env bash
#
# Локальный прогон тестов hw-agent против размещённой модели (vllm на agent GPU VM).
#
# Что делает скрипт:
#   1. проверяет, что вы задали переменную окружения API_KEY (ваш ключ к модели);
#   2. запускает agent GPU VM (она по умолчанию выключена ради экономии);
#   3. ждёт, пока модель станет доступна по сети;
#   4. запускает тесты (pytest).
#
# Использование:
#   export API_KEY="sk-vllm-..."        # ваш ключ
#   ./run_tests.sh                      # прогнать все тесты (tests/)
#   ./run_tests.sh tests/test_agent.py  # прогнать конкретный файл/папку
#
# Можно переопределить через окружение (есть значения по умолчанию):
#   LLM_BASE_URL  (по умолчанию https://vllm.mrsndmn.party/v1)
#   LLM_MODEL     (по умолчанию Qwen/Qwen3-4B-Instruct-2507)
#   WAIT_TIMEOUT  (сколько секунд ждать доступности модели, по умолчанию 600)

set -euo pipefail

# --- настройки (с дефолтами) ---
: "${LLM_BASE_URL:=https://vllm.mrsndmn.party/v1}"
: "${LLM_MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${WAIT_TIMEOUT:=600}"
START_RUNNER_URL="${START_RUNNER_URL:-https://functions.yandexcloud.net/d4eoj3esc5rokpgn1qt8}"
export LLM_BASE_URL LLM_MODEL API_KEY

# --- 1. проверка API_KEY ---
if [ -z "${API_KEY:-}" ]; then
  echo "ОШИБКА: не задана переменная окружения API_KEY." >&2
  echo "Задайте ваш ключ доступа к модели и повторите запуск, например:" >&2
  echo '  export API_KEY="sk-vllm-..."' >&2
  exit 1
fi

# --- 2. запуск agent GPU VM ---
echo "==> Запускаю agent GPU VM..."
if ! curl -fsS --max-time 30 "$START_RUNNER_URL" >/dev/null 2>&1; then
  echo "    (не удалось дёрнуть функцию запуска — возможно, VM уже запущена; продолжаю)"
fi

# --- 3. ожидание доступности модели ---
echo "==> Жду доступности модели на ${LLM_BASE_URL} (до ${WAIT_TIMEOUT}s)..."
deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
code=000
while :; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
           -H "Authorization: Bearer ${API_KEY}" "${LLM_BASE_URL}/models" || echo 000)
  if [ "$code" = "200" ]; then
    echo "==> Модель доступна."
    break
  fi
  if [ "$code" = "401" ] || [ "$code" = "403" ]; then
    echo "ОШИБКА: модель ответила ${code} — неверный API_KEY." >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ОШИБКА: модель не стала доступна за ${WAIT_TIMEOUT}s (последний код: ${code})." >&2
    echo "VM могла ещё не успеть загрузиться — попробуйте запустить скрипт ещё раз." >&2
    exit 1
  fi
  sleep 5
done

# --- 4. запуск тестов ---
if [ "$#" -eq 0 ]; then
  set -- tests/
fi
echo "==> Запускаю тесты: pytest $*"
PYTHONPATH=. exec pytest "$@"
