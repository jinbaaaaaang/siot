import React, { useState, useRef, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import PoemGeneration from './PoemGeneration.jsx'
import About from './About.jsx'
import Archive from './Archive.jsx'
import Settings from './Settings.jsx'
import Logo from '../assets/logo.svg'

function NavLink({ children, active = false, onClick, textRef, to }) {
    const className = `px-4 py-2 text-[#59538F] font-semibold ${active ? 'opacity-100' : 'opacity-80 hover:opacity-100'} transition-opacity relative`
    
    if (to) {
        return (
            <Link to={to} onClick={onClick} className={className}>
                <span ref={textRef}>{children}</span>
            </Link>
        )
    }
    
    return (
        <button onClick={onClick} className={className}>
            <span ref={textRef}>{children}</span>
        </button>
    )
}

function Main() {
    const location = useLocation()
    const [activeNav, setActiveNav] = useState('시 생성')
    const [underlineStyle, setUnderlineStyle] = useState({ left: 0, width: 0 })
    const textRefs = useRef({})
    const navContainerRef = useRef(null)

    const navLinks = [
        { label: '시옷이란', key: 'about', to: '/app/about' },
        { label: '시 생성', key: 'generation', to: '/app/generation' },
        { label: '시 보관함', key: 'archive', to: '/app/archive' },
    ]
    
    // URL에 따라 활성 탭 설정
    useEffect(() => {
        const path = location.pathname
        if (path === '/app/about') setActiveNav('시옷이란')
        else if (path === '/app/generation') setActiveNav('시 생성')
        else if (path === '/app/archive') setActiveNav('시 보관함')
        else if (path === '/app/settings') setActiveNav('설정')
        else if (path === '/app') setActiveNav('시 생성')
    }, [location.pathname])
    
    // 각 네비게이션에 해당하는 컴포넌트
    const navComponents = {
        'generation': <PoemGeneration key="generation" />,
        'about': <About key="about" />,
        'archive': <Archive key="archive" />,
        'settings': <Settings key="settings" />,
    }

    const updateUnderline = () => {
        const activeTextSpan = textRefs.current[activeNav]
        const container = navContainerRef.current
        
        if (activeTextSpan && container) {
            const textRect = activeTextSpan.getBoundingClientRect()
            const containerRect = container.getBoundingClientRect()
            setUnderlineStyle({
                left: textRect.left - containerRect.left,
                width: textRect.width,
            })
        }
    }

    useEffect(() => {
        const timer = setTimeout(() => {
            updateUnderline()
        }, 0)
        
        window.addEventListener('resize', updateUnderline)
        return () => {
            clearTimeout(timer)
            window.removeEventListener('resize', updateUnderline)
        }
    }, [activeNav])

    return (
        <div className="w-full min-h-screen bg-[#FAF5F1]">
            {/* Header */}
            <header className="w-full px-6 sm:px-8 md:px-10 py-4">
                <div className="flex items-center justify-between">
                    {/* Logo */}
                    <div className="flex items-center gap-2">
                        <img src={Logo} alt="시옷" className="h-15 w-auto" />
                    </div>
                    
                    {/* Center Navigation Links */}
                    <nav ref={navContainerRef} className="relative flex items-center gap-6">
                        {navLinks.map((nav) => (
                            <NavLink
                                key={nav.key}
                                to={nav.to}
                                active={activeNav === nav.label}
                                onClick={() => setActiveNav(nav.label)}
                                textRef={(el) => (textRefs.current[nav.label] = el)}
                            >
                                {nav.label}
                            </NavLink>
                        ))}
                        {/* Animated underline */}
                        <div
                            className="absolute bottom-0 h-0.5 bg-[#59538F] transition-all duration-300 ease-in-out"
                            style={{
                                left: `${underlineStyle.left}px`,
                                width: `${underlineStyle.width}px`,
                            }}
                        />
                    </nav>
                    
                    {/* Settings Link - Right aligned */}
                    <div>
                        <NavLink
                            to="/app/settings"
                            active={activeNav === '설정'}
                            onClick={() => setActiveNav('설정')}
                            textRef={(el) => (textRefs.current['설정'] = el)}
                        >
                            설정
                        </NavLink>
                    </div>
                </div>
            </header>
            
            {/* Main Content Area */}
            <main className="w-full min-h-[calc(100vh-80px)] bg-[#FAF5F1]">
                {navComponents[activeNav === '시 생성' ? 'generation' : activeNav === '설정' ? 'settings' : activeNav === '시옷이란' ? 'about' : 'archive']}
            </main>
        </div>
    )
}

export default Main
