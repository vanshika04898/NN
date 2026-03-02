import { BrowserRouter, Routes, Route } from "react-router-dom";
import {
  ClerkLoaded,
  ClerkLoading,
  RedirectToSignIn,
  useUser,
} from "@clerk/clerk-react";
import Layout from "./components/Layout";
import LandingPage from "./pages/LandingPage";
import VisitorForm from "./features/visitor/VisitorForm";
import AdminDashboard from "./features/admin/Dashboard";
import ReceiptsPage from "./pages/ReceiptsPage";
import Loader from "./components/Loader";
import ScrollToTop from "./components/ScrollToTop"; 
import { ShieldAlert } from "lucide-react";


const ProtectedRoute = ({ children }) => {
  const { isLoaded, isSignedIn } = useUser();

  if (!isLoaded) return <Loader fullScreen={true} />;

  if (!isSignedIn) {
    return (
      <>
        <Loader fullScreen={true} />
        <RedirectToSignIn />
      </>
    );
  }

  return children;
};

const AdminRoute = ({ children }) => {
  const { user, isLoaded, isSignedIn } = useUser();

  if (!isLoaded) return <Loader fullScreen={true} />;

  if (!isSignedIn) {
    return (
      <>
        <Loader fullScreen={true} />
        <RedirectToSignIn />
      </>
    );
  }

  if (user?.publicMetadata?.role !== "admin") {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center text-center p-4">
        <div className="w-20 h-20 bg-red-500/10 rounded-full flex items-center justify-center mb-6">
          <ShieldAlert className="text-red-500 w-10 h-10" />
        </div>
        <h1 className="text-3xl font-black text-white mb-2">Access Denied</h1>
        <p className="text-white/50 max-w-md">
          You do not have permission to view this area.
        </p>
      </div>
    );
  }

  return children;
};

function App() {
  return (
    <>
      <ClerkLoading>
        <Loader fullScreen={true} />
      </ClerkLoading>

      <ClerkLoaded>
        <BrowserRouter>
          <ScrollToTop />

          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<LandingPage />} />

              <Route
                path="/apply"
                element={
                  <ProtectedRoute>
                    <VisitorForm />
                  </ProtectedRoute>
                }
              />

              <Route
                path="/receipts"
                element={
                  <ProtectedRoute>
                    <ReceiptsPage />
                  </ProtectedRoute>
                }
              />

              <Route
                path="/admin"
                element={
                  <AdminRoute>
                    <AdminDashboard />
                  </AdminRoute>
                }
              />
            </Route>
          </Routes>
        </BrowserRouter>
      </ClerkLoaded>
    </>
  );
}

export default App;
