import React from 'react'
import { Link } from 'react-router-dom'

function Footer() {
    return (
        <footer className="w-full py-8 px-6 sm:px-8 md:px-10">
            <div className="max-w-7xl mx-auto">
                <div className="flex flex-col items-center gap-6">
                    {/* 로고/브랜드 */}
                    <div className="text-center">
                        <p className="text-gray-700 text-lg font-medium mb-1">시옷</p>
                        <p className="text-gray-500 text-sm">
                            일상글을 시로 만드는 AI 서비스
                        </p>
                    </div>
                    
                    {/* 링크 */}
                    <div className="flex items-center gap-6 text-sm">
                        <Link 
                            to="/app/about" 
                            className="text-gray-600 hover:text-gray-800 transition-colors"
                        >
                            About
                        </Link>
                        <span className="text-gray-300">•</span>
                        <a 
                            href="https://github.com/jinbaaaaaang" 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-gray-600 hover:text-gray-800 transition-colors"
                        >
                            GitHub
                        </a>
                    </div>
                    
                </div>
            </div>
        </footer>
    )
}

export default Footer

