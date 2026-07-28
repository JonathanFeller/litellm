import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { navigateWithParams } from "../../../navigateWithParams";

export interface AutoRouterDetailRouting {
  autoRouterId: string | null;
  openAutoRouter: (id: string) => void;
  close: () => void;
}

/**
 * Mirrors models-and-endpoints/detailNavigation.ts: the selected row lives in the URL so a
 * detail view is linkable and the back button works, rather than being trapped in a modal.
 */
export function useAutoRouterDetailRouting(): AutoRouterDetailRouting {
  const searchParams = useSearchParams();

  const openAutoRouter = useCallback((id: string) => {
    navigateWithParams((params) => {
      params.set("autoRouter", id);
    });
  }, []);

  const close = useCallback(() => {
    navigateWithParams((params) => {
      params.delete("autoRouter");
    });
  }, []);

  return {
    autoRouterId: searchParams?.get("autoRouter") ?? null,
    openAutoRouter,
    close,
  };
}
