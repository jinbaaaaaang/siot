import React, { useState, useEffect, useRef } from 'react'

// API URL 설정 (.env 파일에서 읽기)
// - SOLAR 모델: Colab URL 사용 (VITE_COLAB_API_URL)
// - koGPT2 모델: 로컬 URL 사용 (VITE_API_URL 또는 localhost)
const COLAB_API_URL = import.meta.env.VITE_COLAB_API_URL || ''  // Colab ngrok URL
const LOCAL_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/poem/generate'
const STORAGE_KEY = 'saved_poems'
const SETTINGS_KEY = 'app_settings'

// 환경 변수 디버깅 (개발 모드에서만)
if (import.meta.env.DEV) {
    console.log('🔍 환경 변수 확인:', {
        VITE_COLAB_API_URL: import.meta.env.VITE_COLAB_API_URL || '(없음)',
        VITE_API_URL: import.meta.env.VITE_API_URL || '(없음)',
        COLAB_API_URL: COLAB_API_URL || '(없음)',
        LOCAL_API_URL
    })
}

// API 호출 함수
const callPoemAPI = async (apiBaseUrl, endpoint, requestBody, signal) => {
    // apiBaseUrl이 절대 URL인지 확인 (http:// 또는 https://로 시작해야 함)
    if (!apiBaseUrl || (!apiBaseUrl.startsWith('http://') && !apiBaseUrl.startsWith('https://'))) {
        const errorMsg = `❌ 잘못된 API URL: ${apiBaseUrl}. 절대 URL이 필요합니다 (http:// 또는 https://로 시작).`
        console.error(errorMsg)
        throw new Error(errorMsg)
    }
    
    // apiBaseUrl 끝의 슬래시 제거
    const cleanBaseUrl = apiBaseUrl.replace(/\/$/, '')
    // endpoint 앞의 슬래시 추가 (없으면)
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`
    const apiUrl = `${cleanBaseUrl}${cleanEndpoint}`
    
    // 최종 URL이 절대 URL인지 다시 확인
    if (!apiUrl.startsWith('http://') && !apiUrl.startsWith('https://')) {
        const errorMsg = `❌ 잘못된 최종 API URL: ${apiUrl}. 절대 URL이 필요합니다.`
        console.error(errorMsg)
        throw new Error(errorMsg)
    }
    
    // 헤더 설정
    const headers = {
        'Content-Type': 'application/json',
    }
    
    // ngrok 무료 버전 경고 페이지 우회
    if (apiBaseUrl.includes('ngrok-free.dev') || apiUrl.includes('ngrok-free.dev')) {
        headers['ngrok-skip-browser-warning'] = 'true'
    }
    
    console.log('📤 API 요청:', {
        url: apiUrl,
        method: 'POST',
        headers,
        body: requestBody,
        isAbsoluteUrl: apiUrl.startsWith('http://') || apiUrl.startsWith('https://')
    })
    
    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(requestBody),
            signal: signal,
            mode: 'cors',  // CORS 명시적 설정
            credentials: 'omit'  // 쿠키 전송 안 함
        })
        
        console.log('📥 API 응답:', {
            status: response.status,
            statusText: response.statusText,
            url: response.url,
            ok: response.ok,
            headers: Object.fromEntries(response.headers.entries())
        })
        
        return response
    } catch (fetchError) {
        console.error('❌ fetch 오류 상세:', {
            name: fetchError.name,
            message: fetchError.message,
            stack: fetchError.stack,
            url: apiUrl
        })
        throw fetchError
    }
}

// 커스텀 드롭다운 컴포넌트
function CustomDropdown({ value, onChange, options, placeholder, disabled }) {
    const [isOpen, setIsOpen] = useState(false)
    const dropdownRef = useRef(null)

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsOpen(false)
            }
        }

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside)
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside)
        }
    }, [isOpen])

    const selectedOption = options.find(opt => opt.value === value) || { label: placeholder }

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                type="button"
                onClick={() => !disabled && setIsOpen(!isOpen)}
                disabled={disabled}
                className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-400 focus:border-gray-600 text-sm text-left flex items-center justify-between cursor-pointer hover:border-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <span className={value ? 'text-gray-800' : 'text-gray-600'}>
                    {selectedOption.label}
                </span>
                <svg 
                    className={`w-4 h-4 text-gray-600 transition-transform duration-200 ${isOpen ? 'transform rotate-180' : ''}`}
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>
            
            {isOpen && (
                <div className="absolute z-10 w-full mt-1 border border-gray-600 rounded-lg shadow-lg max-h-48 overflow-y-auto bg-white/30 backdrop-blur-xl">
                    {options.map((option) => (
                        <button
                            key={option.value}
                            type="button"
                            onClick={() => {
                                onChange(option.value)
                                setIsOpen(false)
                            }}
                            className={`w-full px-3 py-2 text-left text-sm transition-colors ${
                                value === option.value
                                    ? 'bg-[#79A9E6]/80 text-white'
                                    : 'text-gray-800 hover:bg-white/20'
                            }`}
                        >
                            {option.label}
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}

function PoemGeneration() {
    const [text, setText] = useState('')
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [saved, setSaved] = useState(false)
    
    const [modelType, setModelType] = useState('')  // 'solar' 또는 'kogpt2'
    const [useTrainedModel, setUseTrainedModel] = useState(false)  // 학습된 모델 사용 여부
    const [loadingDots, setLoadingDots] = useState('')
    
    // 설정 로드 (컴포넌트 마운트 시)
    useEffect(() => {
        try {
            const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')
            // 기본 모델 타입 설정
            if (settings.defaultModelType) {
                setModelType(settings.defaultModelType)
                // koGPT2 선택 시 학습된 모델 자동 사용
                if (settings.defaultModelType === 'kogpt2') {
                    setUseTrainedModel(true)
                } else if (settings.defaultModelType === 'solar') {
                    setUseTrainedModel(false)
                }
            }
        } catch (err) {
            console.error('설정 로드 실패:', err)
        }
    }, [])

    useEffect(() => {
        if (!loading) {
            setLoadingDots('')
            return
        }

        const dotSequence = ['', '.', '..', '...']
        let index = 0

        const intervalId = setInterval(() => {
            index = (index + 1) % dotSequence.length
            setLoadingDots(dotSequence[index])
        }, 400)

        return () => clearInterval(intervalId)
    }, [loading])

    const handleSubmit = async (e) => {
        e.preventDefault()
        
        if (!text.trim()) {
            setError('텍스트를 입력해주세요.')
            return
        }

        setLoading(true)
        setError(null)
        setResult(null)

        try {
            // 타임아웃 설정 (백엔드 타임아웃과 맞춤: 300초 = 5분, 첫 요청 시 모델 로딩으로 더 오래 걸릴 수 있음)
            const controller = new AbortController()
            const timeoutId = setTimeout(() => controller.abort(), 330000) // 5.5분 (백엔드 300초 + 여유)
            
            // 요청 본문 구성
            const requestBody = {
                text: text.trim(),
                ...(modelType ? { model_type: modelType } : {}),
                ...(useTrainedModel ? { use_trained_model: true } : {}),
            }
            
            // 모델 타입에 따라 API Base URL 선택
            // SOLAR 모델 선택 시에만 Colab API 사용
            let apiBaseUrl = ''
            const endpoint = '/api/poem/generate'
            
            console.log('🔍 모델 타입 확인:', {
                modelType,
                COLAB_API_URL: COLAB_API_URL || '(없음)',
                LOCAL_API_URL
            })
            
            if (modelType === 'solar') {
                // SOLAR 모델: 반드시 Colab API 사용 (로컬 서버 사용 안 함)
                if (!COLAB_API_URL || COLAB_API_URL.trim() === '') {
                    console.error('❌ SOLAR 모델을 사용하려면 코랩 URL이 필요합니다!')
                    console.error('현재 COLAB_API_URL:', COLAB_API_URL)
                    console.warn('💡 .env 파일에 VITE_COLAB_API_URL을 설정하고 프론트엔드를 재시작하세요.')
                    setError('SOLAR 모델을 사용하려면 코랩 URL이 필요합니다. .env 파일에 VITE_COLAB_API_URL을 설정하고 프론트엔드를 재시작하세요.')
                    setLoading(false)
                    return
                }
                
                // 로컬 서버 URL이 포함되어 있으면 에러
                if (COLAB_API_URL.includes('localhost') || COLAB_API_URL.includes('127.0.0.1') || COLAB_API_URL.includes(':8000')) {
                    console.error('❌ SOLAR 모델은 로컬 서버를 사용할 수 없습니다!')
                    console.error('현재 COLAB_API_URL:', COLAB_API_URL)
                    setError('SOLAR 모델은 코랩 서버만 사용 가능합니다. .env 파일의 VITE_COLAB_API_URL을 코랩 ngrok URL로 설정해주세요.')
                    setLoading(false)
                    return
                }
                
                // 절대 URL인지 확인 (http:// 또는 https://로 시작해야 함)
                const trimmedUrl = COLAB_API_URL.trim()
                if (!trimmedUrl.startsWith('http://') && !trimmedUrl.startsWith('https://')) {
                    console.error('❌ COLAB_API_URL이 절대 URL이 아닙니다!')
                    console.error('현재 COLAB_API_URL:', trimmedUrl)
                    console.error('💡 절대 URL이 필요합니다 (예: https://xxxx.ngrok-free.dev)')
                    setError('COLAB_API_URL이 올바른 형식이 아닙니다. https://로 시작하는 절대 URL이 필요합니다.')
                    setLoading(false)
                    return
                }
                
                apiBaseUrl = trimmedUrl
                console.log('🌐 SOLAR 모델 선택됨 → Colab API로 요청 전송')
                console.log('📡 Colab 서버 URL:', apiBaseUrl)
                console.log('⚠️ 주의: 로컬 서버가 아닌 Colab 서버로 요청이 전송됩니다!')
                console.log('✅ 로컬 서버로 요청하지 않습니다!')
                console.log('✅ 절대 URL 확인됨:', apiBaseUrl.startsWith('http://') || apiBaseUrl.startsWith('https://'))
            } else {
                // koGPT2 모델 또는 모델 미선택: 로컬 서버 사용
                apiBaseUrl = LOCAL_API_URL.replace('/api/poem/generate', '')  // base URL만 추출
                if (!apiBaseUrl || apiBaseUrl === LOCAL_API_URL) {
                    apiBaseUrl = 'http://localhost:8000'
                }
                console.log('💻 koGPT2 모델 선택됨 → 로컬 서버 사용:', apiBaseUrl)
            }
            
            // API 호출 (SOLAR 모델이면 Colab, 아니면 로컬)
            console.log('🚀 API 요청 시작:', { 
                modelType, 
                apiBaseUrl, 
                endpoint,
                fullUrl: `${apiBaseUrl}${endpoint}`,
                isColab: modelType === 'solar'
            })
            const response = await callPoemAPI(apiBaseUrl, endpoint, requestBody, controller.signal)
            
            clearTimeout(timeoutId)

            let data
            try {
                data = await response.json()
            } catch (jsonError) {
                // JSON 파싱 실패 시 텍스트 응답 사용
                const text = await response.text()
                setError(`서버 오류: ${response.status} ${response.statusText}${text ? ` - ${text.substring(0, 200)}` : ''}`)
                return
            }

            if (!response.ok) {
                // 백엔드에서 반환하는 상세 에러 메시지 표시
                const errorMessage = data.detail || data.message || `서버 오류: ${response.status} ${response.statusText}`
                setError(errorMessage)
                return
            }

            if (data.success) {
                console.log('✅ 시 생성 성공!', {
                    modelType: modelType === 'solar' ? 'SOLAR (Colab)' : 'koGPT2 (로컬)',
                    poem_length: data.poem?.length || 0,
                    keywords: data.keywords,
                    emotion: data.emotion
                })
                
                if (modelType === 'solar') {
                    console.log('🎉 Colab에서 생성된 시를 프론트엔드로 받아왔습니다!')
                }
                
                setResult(data)
                setSaved(false)
                
                // 자동 저장 기능 (설정에서 활성화된 경우)
                try {
                    const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')
                    if (settings.autoSave !== false) {  // 기본값은 true
                        handleSavePoem(data)
                    }
                } catch (err) {
                    console.error('자동 저장 설정 확인 실패:', err)
                }
            } else {
                setError(data.message || '시 생성에 실패했습니다.')
            }
        } catch (err) {
            console.error('❌ API 호출 오류:', err)
            
            if (err.name === 'AbortError') {
                setError('시 생성 시간이 너무 오래 걸려 중단되었습니다. 첫 요청은 모델 로딩으로 5분 이상 걸릴 수 있습니다. 잠시 후 다시 시도해주세요.')
            } else if (err.name === 'TypeError' && err.message.includes('fetch')) {
                // 상세한 에러 정보 로깅
                console.error('❌ 네트워크 오류 상세:', {
                    name: err.name,
                    message: err.message,
                    stack: err.stack,
                    modelType,
                    COLAB_API_URL,
                    apiBaseUrl: modelType === 'solar' ? COLAB_API_URL : 'localhost:8000'
                })
                
                let errorMsg = '서버에 연결할 수 없습니다.\n\n'
                
                if (modelType === 'solar' && COLAB_API_URL) {
                    errorMsg += '🔍 디버깅 정보:\n'
                    errorMsg += `   - 요청 URL: ${COLAB_API_URL}/api/poem/generate\n`
                    errorMsg += `   - 에러: ${err.message}\n\n`
                    errorMsg += '💡 해결 방법:\n'
                    errorMsg += '1. 브라우저에서 ngrok URL 직접 접속:\n'
                    errorMsg += `   ${COLAB_API_URL}\n`
                    errorMsg += '   → "Visit Site" 버튼 클릭 (ngrok 경고 페이지 우회)\n\n'
                    errorMsg += '2. 브라우저 콘솔(F12)에서 Network 탭 확인:\n'
                    errorMsg += '   - 요청이 실제로 전송되었는지 확인\n'
                    errorMsg += '   - CORS 오류가 있는지 확인\n\n'
                    errorMsg += '3. 서버 상태 확인:\n'
                    errorMsg += `   curl -H "ngrok-skip-browser-warning: true" ${COLAB_API_URL}/health\n`
                } else if (modelType !== 'solar') {
                    errorMsg += '로컬 백엔드 서버가 실행 중인지 확인해주세요.\n'
                    errorMsg += '서버 실행: cd backend && python -m uvicorn app.main:app --reload'
                } else {
                    errorMsg += '백엔드 서버가 실행 중인지 확인해주세요.'
                }
                
                setError(errorMsg)
            } else {
                setError(`오류가 발생했습니다: ${err.message || '알 수 없는 오류'}`)
            }
        } finally {
            setLoading(false)
        }
    }

    const handleReset = () => {
        setText('')
        setResult(null)
        setError(null)
        setSaved(false)
        setModelType('')
        setUseTrainedModel(false)
        setLoadingDots('')
    }

    const handleSavePoem = (poemResult = null) => {
        // poemResult가 없으면 현재 result 사용 (수동 저장)
        const dataToSave = poemResult || result
        if (!dataToSave || !dataToSave.poem) return

        const poemData = {
            id: Date.now().toString(),
            poem: dataToSave.poem,
            keywords: dataToSave.keywords || [],
            emotion: dataToSave.emotion || '',
            emotion_confidence: dataToSave.emotion_confidence || 0,
            originalText: text.trim(),
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        }

        try {
            const savedPoems = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            savedPoems.unshift(poemData) // 최신 시가 맨 위에 오도록
            localStorage.setItem(STORAGE_KEY, JSON.stringify(savedPoems))
            setSaved(true)
        } catch (err) {
            console.error('시 저장 실패:', err)
            setError('시 저장에 실패했습니다.')
        }
    }

    return (
        <div className="px-6 sm:px-8 md:px-10 pt-4 sm:pt-6 md:pt-8 pb-4 sm:pb-6 md:pb-8 max-w-4xl mx-auto">
            <h2 className="text-2xl sm:text-3xl font-semibold text-gray-800 mb-3">
                시 생성
            </h2>

            <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                    <label 
                        htmlFor="text-input" 
                        className="block text-sm font-medium text-gray-800 mb-4"
                    >
                        일상글을 입력해주세요
                    </label>
                    <textarea
                        id="text-input"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="오늘 하루는 어떤 하루였나요? 당신의 일상을 들려주세요..."
                        className="w-full px-4 py-3 border border-gray-600 rounded-lg focus:outline-none focus:border-gray-600 resize-none text-gray-800"
                        rows="12"
                        disabled={loading}
                    />
                </div>

                {/* 모델 선택 */}
                <div className="rounded-lg p-4 border border-gray-600 bg-transparent">
                    <label className="block text-sm font-medium text-gray-800 mb-3">
                        모델 선택
                    </label>
                    <div className="flex gap-3">
                        <button
                            type="button"
                            onClick={() => {
                                setModelType('solar')
                                setUseTrainedModel(false)  // SOLAR 선택 시 기본 모델 사용
                            }}
                            disabled={loading}
                            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                                modelType === 'solar'
                                    ? 'bg-[#79A9E6] text-white'
                                    : 'bg-transparent border border-gray-600 text-gray-800 hover:bg-gray-50'
                            } disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                            SOLAR (GPU)
                            <div className="text-xs mt-1 opacity-80">
                                {COLAB_API_URL ? 'Colab 연동' : '로컬 서버'}
                            </div>
                        </button>
                        <button
                            type="button"
                            onClick={() => {
                                setModelType('kogpt2')
                                setUseTrainedModel(true)  // koGPT2 선택 시 학습된 모델 자동 사용
                            }}
                            disabled={loading}
                            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                                modelType === 'kogpt2'
                                    ? 'bg-[#79A9E6] text-white'
                                    : 'bg-transparent border border-gray-600 text-gray-800 hover:bg-gray-50'
                            } disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                            koGPT2 (CPU)
                            <div className="text-xs mt-1 opacity-80">학습된 모델 사용</div>
                        </button>
                    </div>
                    {!modelType && (
                        <p className="text-xs text-gray-600 mt-2">
                            모델을 선택하지 않으면 자동으로 GPU/CPU를 감지하여 선택됩니다.
                        </p>
                    )}
                    
                    {/* 학습된 모델 사용 상태 표시 */}
                    {modelType === 'kogpt2' && (
                        <div className="mt-4 pt-4 border-t border-gray-300">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-medium text-gray-800">
                                    ✅ 학습된 모델 사용 중
                                </span>
                            </div>
                            <p className="text-xs text-gray-600 mt-1">
                                Colab에서 학습한 모델로 시를 생성합니다. 산문의 의미를 이해하고 시로 변환합니다.
                            </p>
                        </div>
                    )}
                </div>

                <div className="flex gap-3">
                    <button
                        type="submit"
                        disabled={loading || !text.trim()}
                        className="w-40 px-6 py-3 bg-transparent border border-gray-800 text-gray-800 rounded-lg font-medium hover:bg-gray-50 hover:border-gray-600 disabled:cursor-not-allowed transition-colors text-center"
                    >
                        {loading ? `시 생성 중${loadingDots}` : '시 생성하기'}
                    </button>
                    
                    {result && (
                        <button
                            type="button"
                            onClick={handleReset}
                            className="px-6 py-3 bg-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-400 transition-colors"
                        >
                            다시 작성
                        </button>
                    )}
                </div>
            </form>

            {error && (
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                    {error}
                </div>
            )}

            {result && (
                <div className="mt-8 space-y-6">
                    {/* 키워드 */}
                    {result.keywords && result.keywords.length > 0 && (
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">
                                추출된 키워드
                            </h3>
                            <div className="flex flex-wrap gap-2">
                                {result.keywords.map((keyword, index) => (
                                    <span
                                        key={index}
                                        className="px-3 py-1 bg-white border border-gray-600 text-gray-800 rounded-full text-sm"
                                    >
                                        {keyword}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* 감정 */}
                    {result.emotion && (
                        <div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">
                                감정 분석
                            </h3>
                            <div className="flex items-center gap-3">
                                <span className="px-4 py-2 bg-white border border-gray-600 text-gray-800 rounded-lg font-medium">
                                    {result.emotion}
                                </span>
                                <span className="text-sm text-gray-600">
                                    (신뢰도: {(result.emotion_confidence * 100).toFixed(1)}%)
                                </span>
                            </div>
                        </div>
                    )}

                    {/* 생성된 시 */}
                    {result.poem && (
                        <div>
                            <div className="flex items-center justify-between mb-3">
                                <h3 className="text-lg font-semibold text-gray-800">
                                생성된 시
                            </h3>
                                <button
                                    type="button"
                                    onClick={handleSavePoem}
                                    disabled={saved}
                                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                                        saved
                                            ? 'bg-green-100 text-green-700 cursor-not-allowed'
                                            : 'bg-[#79A9E6] text-white hover:bg-[#5A8FD6]'
                                    }`}
                                >
                                    {saved ? '✓ 보관함에 저장됨' : '보관함에 저장'}
                                </button>
                            </div>
                            <div className="p-6 bg-transparent border border-gray-600 rounded-lg">
                                <div className="whitespace-pre-line text-gray-800 leading-relaxed">
                                    {result.poem}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default PoemGeneration
