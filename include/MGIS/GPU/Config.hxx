/*!
 * \file   MGIS/GPU/Config.hxx
 * \brief
 * \author Thomas Helfer
 * \date   05/02/2026
 */

#ifndef LIB_MGIS_GPU_CONFIG_HXX
#define LIB_MGIS_GPU_CONFIG_HXX

#include "MGIS/Config.hxx"

#if defined _WIN32 || defined _WIN64 || defined __CYGWIN__
#if defined MGISGPU_EXPORTS
#define MGIS_GPU_EXPORT MGIS_VISIBILITY_EXPORT
#else
#ifndef MGIS_STATIC_BUILD
#define MGIS_GPU_EXPORT MGIS_VISIBILITY_IMPORT
#else
#define MGIS_GPU_EXPORT
#endif
#endif
#else
#define MGIS_GPU_EXPORT MGIS_VISIBILITY_EXPORT
#endif /* */

#endif /* LIB_MGIS_GPU_CONFIG_HXX */
