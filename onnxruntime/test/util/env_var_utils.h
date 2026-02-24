// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <charconv>
#include <locale>
#include <sstream>
#include <optional>
#include <iostream>
#include <cstdlib>

template <typename T>
constexpr bool ParseWithFromChars = !std::is_same_v<bool, T> && (std::is_integral_v<T> || std::is_floating_point_v<T>);

/**
 * Tries to parse a value from an entire string.
 * If successful, sets `value` and returns true. Otherwise, does not modify `value` and returns false.
 */

template <typename T>
std::enable_if_t<ParseWithFromChars<T>, bool>
TryParseStringWithClassicLocale(std::string_view str, T& value) {
  T parsed_value{};

  std::from_chars_result conversion_result{};
  if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    // For unsigned integral types, also handle hex values, i.e., those beginning with "0x".
    // std::from_chars() does not accept the "0x" prefix.
    const bool has_hex_prefix = str.size() >= 2 &&
                                str[0] == '0' &&
                                (str[1] == 'x' || str[1] == 'X');

    if (has_hex_prefix) {
      str = str.substr(2);
    }

    const int base = has_hex_prefix ? 16 : 10;
    conversion_result = std::from_chars(str.data(), str.data() + str.size(), parsed_value, base);
  } else {
    conversion_result = std::from_chars(str.data(), str.data() + str.size(), parsed_value);
  }

  if (conversion_result.ec != std::errc{}) {
    return false;
  }

  if (conversion_result.ptr != str.data() + str.size()) {
    return false;
  }

  value = parsed_value;
  return true;
}

template <typename T>
std::enable_if_t<!ParseWithFromChars<T>, bool>
TryParseStringWithClassicLocale(std::string_view str, T& value) {
  // don't allow leading whitespace
  if (!str.empty() && std::isspace(str[0], std::locale::classic())) {
    return false;
  }

  std::istringstream is{std::string{str}};
  is.imbue(std::locale::classic());
  T parsed_value{};

  const bool parse_successful =
      is >> parsed_value &&
      is.get() == std::istringstream::traits_type::eof();  // don't allow trailing characters
  if (!parse_successful) {
    return false;
  }

  value = std::move(parsed_value);
  return true;
}

inline bool TryParseStringWithClassicLocale(std::string_view str, std::string& value) {
  value = str;
  return true;
}

inline bool TryParseStringWithClassicLocale(std::string_view str, bool& value) {
  if (str == "0" || str == "False" || str == "false") {
    value = false;
    return true;
  }

  if (str == "1" || str == "True" || str == "true") {
    value = true;
    return true;
  }

  return false;
}

template <typename T>
std::optional<T> ParseEnvironmentVariable(const std::string& name) {
#ifdef _WIN32
  char* env_var = nullptr;
  size_t sz = 0;
  _dupenv_s(&env_var, &sz, name.c_str());

  // Check if environment variable exists
  if (env_var == nullptr) {
    return {};
  }

  const std::string value_str(env_var);
  // Free the memory allocated by _dupenv_s
  free(env_var);
#else
  // On non-Windows platforms, use getenv
  const char* env_var = std::getenv(name.c_str());

  // Check if environment variable exists
  if (env_var == nullptr) {
    return {};
  }

  const std::string value_str(env_var);
#endif

  if (value_str.empty()) {
    return {};
  }

  T parsed_value;
  bool parse_res = TryParseStringWithClassicLocale(value_str, parsed_value);
  if (!parse_res) {
    std::cerr << "Failed to parse environment variable - name: \"" + name + "\", value: \"" + value_str + "\"" << std::endl;
  }

  return parsed_value;
}

/**
 * Parses an environment variable value or returns the given default if unavailable.
 */
template <typename T>
T ParseEnvironmentVariableWithDefault(const std::string& name, const T& default_value) {
  const auto parsed = ParseEnvironmentVariable<T>(name);
  if (parsed.has_value()) {
    return *parsed;
  }

  return default_value;
}
