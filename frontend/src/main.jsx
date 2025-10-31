import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './index.css'

import AppLayout from './layouts/AppLayout.jsx'
import Landing from './pages/Landing.jsx'
import Main from './pages/Main.jsx'

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <Landing /> },
      { path: 'app', element: <Main /> },
      { path: 'app/about', element: <Main /> },
      { path: 'app/generation', element: <Main /> },
      { path: 'app/archive', element: <Main /> },
      { path: 'app/settings', element: <Main /> },
    ],
  },
])

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
