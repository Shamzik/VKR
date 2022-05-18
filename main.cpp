#include <stxxl/queue>

#include <array>
#include <chrono>
#include <deque>
#include <forward_list>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <set>
#include <thread>

template<typename IntegralType = unsigned long long, unsigned FunctionsCount = 3, typename ValueType = IntegralType, typename QueueType = stxxl::queue<ValueType>, IntegralType InitValue = 1>
class Queue0 {
    std::array<QueueType, FunctionsCount> m_queues;
    std::array<IntegralType, FunctionsCount> m_a, m_b;
    ValueType m_front;

public:
    typedef ValueType ValueType;
    static constexpr IntegralType initValue = InitValue;

    template<typename... Pairs, std::enable_if_t<sizeof...(Pairs) == FunctionsCount, bool> = true>
    explicit
    Queue0(Pairs... p) : m_a{static_cast<IntegralType>(p.first)...}, m_b{static_cast<IntegralType>(p.second)...},
                         m_front{InitValue} {}

    template<typename OtherIntegral>
    explicit Queue0(const std::array<std::pair<OtherIntegral, OtherIntegral>, FunctionsCount>& functions) : m_front{InitValue} {
        for (std::size_t i = 0; i < FunctionsCount; ++i) {
            m_a[i] = static_cast<IntegralType>(functions[i].first);
            m_b[i] = static_cast<IntegralType>(functions[i].second);
        }
    }

    ValueType Process() {
        auto value = m_front;
        std::size_t toPop = 0;
        for (std::size_t i = 0; i != FunctionsCount; ++i) {
            auto y = value * m_a[i] + m_b[i];
            m_queues[i].push(y);
            if (m_queues[i].front() < m_queues[toPop].front()) {
                toPop = i;
            }
        }
        m_front = m_queues[toPop].front();
        m_queues[toPop].pop();
        for (std::size_t i = toPop + 1; i != FunctionsCount; ++i) {
            if (m_queues[i].front() == m_queues[toPop].front()) {
                m_front, m_queues[i].front();
                m_queues[i].pop();
            }
        }
        return value;
    }
};

template<typename MyQueue = Queue0<>, std::size_t DeltaMax = 100, std::size_t ValueLimit = 1'000'000'000>
void main4(unsigned long long a2 = 0, unsigned long long a3 = 2, unsigned long long a6 = 3) {
    std::string id = "main4_";
    id += std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) + "_";
    std::cout << "ID: " << id << std::endl;
    id += std::to_string(a2) + "_" + std::to_string(a3) + "_" + std::to_string(a6);
    std::ofstream log(id);
    constexpr std::size_t deltaMax = DeltaMax;
    constexpr std::size_t valueLimit = ValueLimit;
    auto startTimePoint = std::chrono::system_clock::now();
    MyQueue queue(std::make_pair(2, a2), std::make_pair(3, a3), std::make_pair(6, a6));
    std::array<std::vector<unsigned long long>, deltaMax> popped;
    for (std::size_t deltaM1 = 0; deltaM1 < deltaMax; ++deltaM1) {
        popped[deltaM1].resize(deltaM1 + 1);
    }
    for (auto v = queue.Process(); v < valueLimit; v = queue.Process()) {
        //std::cout << v << std::endl;
        for (std::size_t deltaM1 = 0; deltaM1 < deltaMax; ++deltaM1) {
            ++popped[deltaM1][v % (deltaM1 + 1)];
        }
    }
    log << "% " << std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
    std::set<std::tuple<double, std::size_t, std::size_t>> best;
    for (std::size_t deltaM1 = 0; deltaM1 < deltaMax; ++deltaM1) {
        for (std::size_t r = 0; r != deltaM1 + 1; ++r) {
            best.emplace(popped[deltaM1][r] / static_cast<double>((valueLimit + deltaM1 - r) / (deltaM1 + 1)), deltaM1 + 1, r);
            if (best.size() > 10) {
                best.erase(best.begin());
            }
        }
    }
    for (auto [density, delta, r] : best) {
        log << density << "\t& " << delta << "\t& " << r << " \\\\" << std::endl;
    }
}

typedef unsigned long long Integral;

#define COMPARABLE_AS_OP(LA, RA, OP) \
template<typename Other>             \
auto operator OP(Other ra) const {   \
    auto other = std::remove_reference_t<decltype(*this)>(ra);      \
    return LA OP RA;                 \
}

#define COMPARABLE_AS(LA, RA) \
COMPARABLE_AS_OP(LA, RA, ==)  \
COMPARABLE_AS_OP(LA, RA, !=)  \
COMPARABLE_AS_OP(LA, RA, <=)  \
COMPARABLE_AS_OP(LA, RA, >=)  \
COMPARABLE_AS_OP(LA, RA, <)   \
COMPARABLE_AS_OP(LA, RA, >)

template<typename Integral>
Integral GCD(Integral a, Integral b) {
    while (a != 0) {
        Integral t = a;
        a = b % a;
        b = t;
    }
    return b;
}

template<typename Integral>
Integral LCM(Integral a, Integral b) {
    return a / GCD(a, b) * b;
}

class SafeIntegral {
    static constexpr Integral MAX_VALUE = ~Integral(0);

    Integral value;

public:
    explicit SafeIntegral(Integral value = Integral()) : value(value) {}

    SafeIntegral& operator+=(SafeIntegral other) {
        if (MAX_VALUE - value >= other.value) {
            value += other.value;
            return *this;
        } else {
            throw std::overflow_error("unsigned long long overflowed");
        }
    }

    SafeIntegral operator+(SafeIntegral other) const {
        SafeIntegral res = *this;
        return res += other;
    }

    SafeIntegral& operator-=(SafeIntegral other) {
        value -= other.value;
        return *this;
    }

    SafeIntegral operator-(SafeIntegral other) const {
        SafeIntegral res = *this;
        return res -= other;
    }

    SafeIntegral& operator*=(SafeIntegral other) {
        if (value == 0 || MAX_VALUE / value >= other.value) {
            value *= other.value;
            return *this;
        } else {
            throw std::overflow_error("unsigned long long overflowed");
        }
    }

    SafeIntegral operator*(SafeIntegral other) const {
        SafeIntegral res = *this;
        return res *= other;
    }

    SafeIntegral& operator/=(SafeIntegral other) {
        value /= other.value;
        return *this;
    }

    SafeIntegral operator/(SafeIntegral other) const {
        SafeIntegral res = *this;
        return res /= other;
    }

    SafeIntegral& operator%=(SafeIntegral other) {
        value %= other.value;
        return *this;
    }

    SafeIntegral operator%(SafeIntegral other) const {
        SafeIntegral res = *this;
        return res %= other;
    }

    COMPARABLE_AS(value, other.value)

    [[nodiscard]] Integral toIntegral() const {
        return value;
    }

    explicit operator Integral() const {
        return value;
    }
};

class Rational {
    SafeIntegral numerator = SafeIntegral(0);
    SafeIntegral denominator = SafeIntegral(1);

    static SafeIntegral Reduce(SafeIntegral& a, SafeIntegral& b) {
        SafeIntegral d = GCD(a, b);
        a /= d;
        b /= d;
        return d;
    }

public:
    Rational() = default;
    explicit Rational(Integral value) : numerator(value) {}
    Rational(Integral numerator, Integral denominator) : numerator(numerator), denominator(denominator) {}
    explicit Rational(SafeIntegral numerator, SafeIntegral denominator = SafeIntegral(1)) : numerator(numerator), denominator(denominator) {}

    Rational& operator++() {
        numerator += denominator;
        return *this;
    }

    Rational operator+=(Rational other) {
        auto d = Reduce(denominator, other.denominator);
        numerator *= other.denominator;
        other.numerator *= denominator;
        numerator += other.numerator;
        Reduce(numerator, d);
        denominator *= d;
        denominator *= other.denominator;
        return *this;
    }

    Rational operator+(Rational other) const {
        Rational res = *this;
        return res += other;
    }

    Rational operator-=(Rational other) {
        auto d = Reduce(denominator, other.denominator);
        numerator *= other.denominator;
        other.numerator *= denominator;
        numerator -= other.numerator;
        Reduce(numerator, d);
        denominator *= d;
        denominator *= other.denominator;
        return *this;
    }

    Rational operator-(Rational other) const {
        Rational res = *this;
        return res -= other;
    }

    Rational operator/=(Rational other) {
        Reduce(numerator, other.numerator);
        Reduce(denominator, other.denominator);
        numerator *= other.denominator;
        denominator *= other.numerator;
        return *this;
    }

    Rational operator/(Rational other) const {
        Rational res = *this;
        return res /= other;
    }

    Rational operator*=(Rational other) {
        Reduce(numerator, other.denominator);
        Reduce(denominator, other.numerator);
        numerator *= other.numerator;
        denominator *= other.denominator;
        return *this;
    }

    Rational operator*(Rational other) const {
        Rational res = *this;
        return res *= other;
    }

    COMPARABLE_AS(numerator * other.denominator, denominator * other.numerator)

    static Rational Ceil(Rational r) {
        return Rational(r.numerator / r.denominator + SafeIntegral(r.numerator % r.denominator != 0 ? 1 : 0));
    }

    static Rational Floor(Rational r) {
        return Rational(r.numerator / r.denominator);
    }

    static bool IsInteger(Rational rational) {
        return rational.denominator == 1;
    }

    explicit operator SafeIntegral() const {
        return numerator / denominator;
    }

    [[nodiscard]] SafeIntegral GetNumerator() const {
        return numerator;
    }

    [[nodiscard]] SafeIntegral GetDenominator() const {
        return denominator;
    }

    [[nodiscard]] static SafeIntegral GetNumerator(Rational r) {
        return r.numerator;
    }

    [[nodiscard]] static SafeIntegral GetDenominator(Rational r) {
        return r.denominator;
    }
};

Rational operator/(Integral x, Rational other) {
    auto res = Rational(x);
    return res /= other;
}

template<Integral N>
struct Trie {
    Rational first;
    std::vector<Trie<N - 1>> rest;

    Trie() = default;
    Trie(Rational first, std::vector<Trie<N - 1>>&& rest) : first(first), rest(std::move(rest)) {}

    static std::vector<Trie> Search(Rational sum = Rational(1), Rational start = Rational(2)) {
        auto mean = sum / Rational(N);
        std::vector<Trie<N>> res;
        for (Rational a = std::max(Rational::Floor(1 / sum) + Rational(1), start); 1 / a > mean; ++a) {
            auto rest = Trie<N - 1>::Search(sum - 1 / a, a + Rational(1));
            res.emplace_back(a, std::move(rest));
        }
        return std::move(res);
    }
};

template<>
struct Trie<1> {
    Rational first;

    Trie() = default;
    explicit Trie(Rational first) : first(first) {}

    static std::vector<Trie> Search(Rational sum, Rational start) {
        std::vector<Trie<1>> res;
        if (Rational a = 1 / sum; Rational::IsInteger(a) && a >= start) {
            res.emplace_back(a);
        }
        return std::move(res);
    }
};

template<Integral N>
void Iterate(const std::vector<Trie<N>>& tries, std::vector<SafeIntegral>& buffer, std::function<void(const std::vector<SafeIntegral>&)>& callback) {
    for (auto& trie : tries) {
        buffer.emplace_back(trie.first);
        Iterate(trie.rest, buffer, callback);
        buffer.pop_back();
    }
}

std::ostream& operator<<(std::ostream& out, const SafeIntegral& v) {
    return out << v.toIntegral();
}

template<>
void Iterate<1>(const std::vector<Trie<1>>& tries, std::vector<SafeIntegral>& buffer, std::function<void(const std::vector<SafeIntegral>&)>& callback) {
    for (auto& trie : tries) {
        buffer.emplace_back(trie.first);
        callback(buffer);
        buffer.pop_back();
    }
}

template<Integral N>
void Iterate(const std::vector<Trie<N>>& tries, std::function<void(const std::vector<SafeIntegral>&)> callback) {
    std::vector<SafeIntegral> buffer;
    buffer.reserve(N);
    Iterate<N>(tries, buffer, callback);
}

template<Integral FunctionsCount = 4>
void main2() {
    auto startTimePoint = std::chrono::system_clock::now();
    auto trie = Trie<FunctionsCount>::Search();
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
    Integral counter = 0;
    Integral maxDenominator = 0;
    Iterate(trie, [&counter, &maxDenominator](const std::vector<SafeIntegral>& v) {
        ++counter;
        if (v.back() > maxDenominator) {
            maxDenominator = v.back().toIntegral();
        }
    });
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
    std::cout << counter << ' ' << maxDenominator << '\n';
}

template<Integral FunctionsCount = 4>
void SearchAndIterate(std::function<void(const std::vector<SafeIntegral>&)> callback) {
    auto startTimePoint = std::chrono::system_clock::now();
    auto trie = Trie<FunctionsCount>::Search();
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
    Iterate(trie, callback);
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
}


template<typename MyQueue, std::size_t ValueLimit = 10'000'000'000>
void maxDensityDeltaLogging(unsigned long long a2 = 0, unsigned long long a3 = 2, unsigned long long a6 = 3) {
    std::string id = "maxDensityDeltaLogging" "_";
//    id += std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
//            std::chrono::system_clock::now().time_since_epoch()).count()) + "_";
    id += std::to_string(a2) + "_" + std::to_string(a3) + "_" + std::to_string(a6);
    std::ofstream log(id);
    auto startTimePoint = std::chrono::system_clock::now();
    MyQueue queue(std::make_pair(2, a2), std::make_pair(3, a3), std::make_pair(6, a6));
    double densityDelta;
    unsigned long long maxDelta = 0;
    unsigned long long v, startDelta;
    unsigned long long prev = 0;
    unsigned long long popped = 0;

    constexpr unsigned long long LOG_STEP = 2;

    unsigned long long nextLogged = LOG_STEP;
    do {
        v = queue.Process();
        auto delta = v - prev;
        double dd = static_cast<double>(popped + 1) / (v + 1) - static_cast<double>(popped) / (prev + 1);
        if (maxDelta == 0 || densityDelta < dd) {
            startDelta = prev;
            maxDelta = delta;
            densityDelta = dd;
        }
        while (v >= nextLogged) {
            if (maxDelta > 0) {
                log << maxDelta << '\t' << startDelta << '\t' << densityDelta << std::endl;
            }
            nextLogged *= LOG_STEP;
            maxDelta = 0;
        }
        prev = v;
        ++popped;
    } while (v < ValueLimit);
    log.close();
    std::cout << "% " << id << ": " << std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
}

template<typename ValueType>
struct WithMultiplicity {
    ValueType value;
    unsigned long long multiplicity;

    explicit WithMultiplicity(ValueType v = 0) : value(v), multiplicity(1) {};

    WithMultiplicity& operator,(WithMultiplicity other) {
        multiplicity += other.multiplicity;
        return *this;
    };
    WithMultiplicity& operator*=(ValueType other) {
        value *= other;
        return *this;
    }

    WithMultiplicity operator*(ValueType other) {
        WithMultiplicity res = *this;
        res *= other;
        return res;
    }

    WithMultiplicity& operator+=(ValueType other) {
        value += other;
        return *this;
    }

    WithMultiplicity operator+(ValueType other) {
        WithMultiplicity res = *this;
        res += other;
        return res;
    }

    COMPARABLE_AS(value, other.value)

    ValueType operator-(ValueType other) {
        return value - other;
    }

    explicit operator ValueType() {
        return value;
    }
};

template<typename... F>
class Processor {
    std::tuple<F...> m_callbacks;

public:
    explicit Processor(F&&... callbacks) : m_callbacks(std::move(callbacks)...) {}

    template<typename QueueType = Queue0<>, std::size_t ValueLimit = 10'000'000'000>
    void Process(QueueType& queue) {
        typename QueueType::ValueType v;
        do {
            v = queue.Process();
            bool needToStop = false;
            std::initializer_list<int> _ = {(needToStop = std::get<F>(m_callbacks)(v) || needToStop, 42)...};
            if (needToStop) {
                break;
            }
        } while (v < ValueLimit);
    }
};

class DensityLogger {
    static constexpr Integral LOG_STEP = 2;

    std::ofstream m_out;
    Integral m_nextLogged = LOG_STEP;
    Integral m_popped = 0;

public:
    explicit DensityLogger(const std::string& filename) : m_out(filename, std::ios::out) {}
    DensityLogger(const DensityLogger&) = default;
    DensityLogger(DensityLogger&&) = default;

    ~DensityLogger() {
        m_out.close();
    }

    template<typename ValueType>
    bool operator()(ValueType v) {
        while (v > m_nextLogged) {
            m_out << m_nextLogged << '\t' << m_popped << '\n';
            m_nextLogged *= LOG_STEP;
        }
        ++m_popped;
        return false;
    }
};

class MaxMultiplicityLogger {
    static constexpr Integral LOG_STEP = 2;

    std::ofstream m_out;
    Integral m_nextLogged = LOG_STEP;
    Integral m_maxMultiplicity = 0;

public:
    explicit MaxMultiplicityLogger(const std::string& filename) : m_out(filename, std::ios::out) {}
    MaxMultiplicityLogger(const MaxMultiplicityLogger&) = default;
    MaxMultiplicityLogger(MaxMultiplicityLogger&&) = default;

    ~MaxMultiplicityLogger() {
        m_out.close();
    }

    template<typename ValueType>
    bool operator()(WithMultiplicity<ValueType> v) {
        while (v > m_nextLogged) {
            m_out << m_nextLogged << '\t' << m_maxMultiplicity << '\n';
            m_nextLogged *= LOG_STEP;
        }
        if (v.multiplicity > m_maxMultiplicity) {
            m_maxMultiplicity = v.multiplicity;
        }
        return false;
    }
};

class MaxDeltaLogger {
    static constexpr Integral LOG_STEP = 2;

    std::ofstream m_out;
    Integral m_nextLogged = LOG_STEP;
    Integral m_maxDelta = 0;
    Integral m_prev = 0;

public:
    explicit MaxDeltaLogger(const std::string& filename) : m_out(filename, std::ios::out) {}
    MaxDeltaLogger(const MaxDeltaLogger&) = default;
    MaxDeltaLogger(MaxDeltaLogger&&) = default;

    ~MaxDeltaLogger() {
        m_out.close();
    }

    template<typename ValueType>
    bool operator()(ValueType v) {
        while (v > m_nextLogged) {
            m_out << m_nextLogged << '\t' << m_maxDelta << '\n';
            m_nextLogged *= LOG_STEP;
        }
        if (m_prev == 0 || v - m_prev > m_maxDelta) {
            m_maxDelta = v - m_prev;
        }
        m_prev = static_cast<Integral>(v);
        return false;
    }
};

template<typename MyQueue = Queue0<Integral, 3, WithMultiplicity<Integral>>, std::size_t ValueLimit = 10'000'000'000>
void main6(Integral a2 = 0, Integral a3 = 2, Integral a6 = 3) {
    std::string id = "_";
    id += std::to_string(MyQueue::initValue) + "_";
    id += std::to_string(a2) + "_" + std::to_string(a3) + "_" + std::to_string(a6);
    MyQueue queue(std::make_pair(2, a2), std::make_pair(3, a3), std::make_pair(6, a6));
    Processor processor(//Serializer(std::string("binary") + id),
            DensityLogger(std::string("popped") + id),
            MaxMultiplicityLogger(std::string("max_multiplicity") + id),
            MaxDeltaLogger(std::string("max_delta") + id));
    processor.Process<MyQueue, ValueLimit>(queue);
}

template<void (*func)(Integral, Integral, Integral)>
void main7() {
    auto startTimePoint = std::chrono::system_clock::now();
    func(0, 2, 3);
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
    func(0, 3, 10);
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
    func(2, 0, 15);
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
    func(1, 0, 2);
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
    func(2, 1, 0);
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
    func(1, 6, 0);
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
}

template<void (*func)(Integral, Integral, Integral)>
void main5() {
    stxxl::config::get_instance()->check_initialized();
    auto startTimePoint = std::chrono::system_clock::now();
    std::vector<std::thread> threads;
    threads.emplace_back(func, 0, 2, 3);
    threads.emplace_back(func, 0, 3, 10);
    threads.emplace_back(func, 2, 0, 15);
    threads.emplace_back(func, 1, 0, 2);
    threads.emplace_back(func, 2, 1, 0);
    threads.emplace_back(func, 1, 6, 0);
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
}

void main8() {
    main5<&main6<Queue0<Integral, 3, WithMultiplicity<Integral>, stxxl::queue<WithMultiplicity<Integral>>, 50>>>();
}

template<Integral FunctionsCount = 3>
void main9() {
    SearchAndIterate<FunctionsCount>([](auto v) {
        std::array<SafeIntegral, FunctionsCount> a;
        for (size_t i = 0; i < FunctionsCount; ++i) {
            a[i] = v[i];
        }
        do {
            std::array<std::pair<SafeIntegral, Rational>, FunctionsCount> functions;
            Rational prefixSum;
            SafeIntegral lcm(1);
            SafeIntegral gcd(0);
            for (std::size_t i = 0; i < FunctionsCount; ++i) {
                Rational m(a[i]);
                functions[i] = {a[i], prefixSum * m};
                lcm = LCM(lcm, functions[i].second.GetDenominator());
                gcd = GCD(gcd, functions[i].second.GetNumerator());
                prefixSum += 1 / m;
            }
            for (auto& [m, b] : functions) {
                b /= Rational(gcd);
                b *= Rational(lcm);
                std::cout << m << "x + " << (SafeIntegral)b << '\t';
            }
            std::cout << '\n';
        } while (std::next_permutation(a.begin(), a.end()));
    });
}

template<std::size_t FunctionsCount = 4, typename MyQueue = Queue0<Integral , FunctionsCount, WithMultiplicity<Integral>>, std::size_t ValueLimit = 10'000'000'000>
void ProcessFunctionSet(const std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount>& functions) {
    std::string id = "_";
    id += std::to_string(MyQueue::initValue);
    for (auto [a, b] : functions) {
        id += "_";
        id += std::to_string(a.toIntegral()) + "_" + std::to_string(b.toIntegral());
    }
    MyQueue queue(functions);
    Processor processor(//Serializer(std::string("binary") + id),
            DensityLogger(std::string("popped") + id),
            MaxMultiplicityLogger(std::string("max_multiplicity") + id),
            MaxDeltaLogger(std::string("max_delta") + id));
    processor.Process<MyQueue, ValueLimit>(queue);
}

template<Integral FunctionsCount = 3, std::size_t Skip = 0, std::size_t ValueLimit = 1'000'000'000, bool CalculateTimings = false, std::size_t ThreadsCount = 6>
void IterateAllFunctionSetsAsync(std::function<void(
        const std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount> &)> func = ProcessFunctionSet<FunctionsCount, Queue0<Integral, FunctionsCount, WithMultiplicity<Integral>>, ValueLimit>) {
    stxxl::config::get_instance()->check_initialized();
    SearchAndIterate<FunctionsCount>([skip = Skip, &func](auto v) mutable {
        if (skip != 0) {
            std::cout << skip << " left to Skip. Skipping...\n";
            --skip;
            return;
        }
        std::array<std::thread, ThreadsCount> threads;
        auto startTimePoint = std::chrono::system_clock::now();
        std::array<SafeIntegral, FunctionsCount> a;
        for (size_t i = 0; i < FunctionsCount; ++i) {
            a[i] = v[i];
        }
        std::size_t i = 0;
        do {
            std::array<std::pair<SafeIntegral, Rational>, FunctionsCount> functions;
            Rational prefixSum;
            SafeIntegral lcm(1);
            SafeIntegral gcd(0);
            for (std::size_t i = 0; i < FunctionsCount; ++i) {
                Rational m(a[i]);
                functions[i] = {a[i], prefixSum * m};
                lcm = LCM(lcm, functions[i].second.GetDenominator());
                gcd = GCD(gcd, functions[i].second.GetNumerator());
                prefixSum += 1 / m;
            }
            std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount> out;
            for (std::size_t i = 0; i < FunctionsCount; ++i) {
                functions[i].second /= Rational(gcd);
                functions[i].second *= Rational(lcm);
                out[i] = {functions[i].first, static_cast<SafeIntegral>(functions[i].second)};
            }
            std::sort(out.begin(), out.end());
            i %= ThreadsCount;
            if (threads[i].joinable()) {
                threads[i].join();
            }
            threads[i++] = std::thread(func, out);
        } while (std::next_permutation(a.begin(), a.end()));
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        if (CalculateTimings) {
            std::cout << "% " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    (std::chrono::system_clock::now() - startTimePoint)).count() << " ms" << std::endl;
        }
    });
}

template<std::size_t FunctionsCount>
std::ostream &
operator<<(std::ostream &out, const std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount> &functionSet) {
    for (auto& [a, b] : functionSet) {
        out << a << "x + " << b << '\t';
    }
    return out;
}

template<bool NeedToProcess = true, std::size_t FunctionsCount = 5, std::size_t ValueLimit = 1'000'000'000, typename QueueType = std::queue<WithMultiplicity<Integral>>, bool CalculateTimings = true, std::size_t ThreadsCount = 4>
void HandleOutput() {
    std::vector<std::pair<double, std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount>>> functionSets;
    std::mutex mutex;
    IterateAllFunctionSetsAsync<FunctionsCount, 0, ValueLimit, CalculateTimings, ThreadsCount>(
            [&functionSets, &mutex](
                    const std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount> &functionSet) {
                if (NeedToProcess) {
                    ProcessFunctionSet<FunctionsCount, Queue0<Integral, FunctionsCount, WithMultiplicity<Integral>, QueueType>, ValueLimit>(
                            functionSet);
                }
                std::string id = "popped_1";
                for (auto[a, b]: functionSet) {
                    id += "_";
                    id += std::to_string(a.toIntegral()) + "_" + std::to_string(b.toIntegral());
                }
                std::ifstream fIn(id);
                Integral prefix, popped;
                double density;
                while (fIn >> prefix >> popped) {
                    density = static_cast<double>(popped) / static_cast<double>(prefix);
                }
                std::scoped_lock lock(mutex);
                functionSets.emplace_back(density, functionSet);
            });
    std::sort(functionSets.begin(), functionSets.end());
    constexpr std::size_t topLimit = 10;
    for (size_t i = 0; i < topLimit; ++i) {
        std::cout << functionSets[i].second << functionSets[i].first << '\n';
    }
    std::cout << "...\n";
    for (size_t i = 0; i < topLimit; ++i) {
        std::cout << functionSets[functionSets.size() - topLimit + i].second
                  << functionSets[functionSets.size() - topLimit + i].first << '\n';
    }
}

template<std::size_t FunctionsCount = 4>
class MultiplicityChecker {
    const std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount>& m_functions;
    bool m_success = true;

public:
    explicit MultiplicityChecker(const std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount> &functions)
            : m_functions(functions) {}

    MultiplicityChecker(MultiplicityChecker && other) noexcept : m_functions(other.m_functions) {
        other.m_success = false;
    }

    ~MultiplicityChecker() {
        if (m_success) {
            std::stringstream output;
            output << "No multiplicity for " << m_functions << std::endl;
            std::cout << output.str();
        }
    }

    template<typename ValueType>
    bool operator()(WithMultiplicity<ValueType> v) {
        if (v.multiplicity > 1) {
            m_success = false;
            return true;
        } else {
            return false;
        }
    }
};

template<std::size_t FunctionsCount = 5, typename... F>
class Callbacks {
    std::tuple<F...> m_callbacks;

public:
    explicit Callbacks(const std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount> &functions)
            : m_callbacks(F(functions)...) {}

    Callbacks(Callbacks &&other) noexcept = default;

    template<typename ValueType>
    bool operator()(ValueType v) {
        bool res = false;
        std::initializer_list<int> _ = {(res = std::get<F>(m_callbacks)(v) || res, 42)...};
        return res;
    }
};

template<std::size_t FunctionsCount = 4, typename MyQueue = Queue0<Integral , FunctionsCount, WithMultiplicity<Integral>, std::queue<WithMultiplicity<Integral>>>, std::size_t ValueLimit = 1'000'000'000, typename CallbackType = MultiplicityChecker<FunctionsCount>>
void ProcessWithCallback(const std::array<std::pair<SafeIntegral, SafeIntegral>, FunctionsCount>& functions) {
    MyQueue queue(functions);
    Processor<CallbackType> processor((CallbackType(functions)));
    processor.template Process<MyQueue, ValueLimit>(queue);
}

template<std::size_t... FunctionsCount>
void SearchForFunctionSetWithoutMultiplicity() {
    std::initializer_list<int> _ = {
            (IterateAllFunctionSetsAsync<FunctionsCount, 0, 10'000'000'000, false, 8>(ProcessWithCallback<FunctionsCount, Queue0<Integral, FunctionsCount, WithMultiplicity<Integral>>>), 42)...};
}

int main() {
    SearchForFunctionSetWithoutMultiplicity<4, 5>();
}
